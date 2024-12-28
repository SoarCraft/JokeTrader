namespace JokeTrader.Services;

using Microsoft.Extensions.Options;
using Torch;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

internal class TrainService : BackgroundService {
    private JokerDataLoader loader { get; }

    private JokerOption option { get; }

    private ILogger<TrainService> logger { get; }

    private JokerTransformer model { get; }

    private AdamW optimizer { get; }

    private BCEWithLogitsLoss bce { get; } = nn.BCEWithLogitsLoss();

    private HuberLoss huber { get; } = nn.HuberLoss();

    private optim.lr_scheduler.LRScheduler scheduler { get; }

    private SummaryWriter writer { get; } = utils.tensorboard.SummaryWriter();

    private int patienceCounter { get; set; }

    private double bestValidationLoss { get; set; } = double.MaxValue;

    public TrainService(JokerDataLoader loader, IOptions<JokerOption> options, ILogger<TrainService> logger) {
        this.loader = loader;
        this.option = options.Value;
        this.logger = logger;

        var featureDim = new SeriesFeatures().ToArray().Length;
        this.model = new(featureDim, this.option.EmbedDim, this.option.NumHeads, this.option.NumLayers);
        this.model.to(this.option.Device);

        this.optimizer = optim.AdamW(this.model.parameters(), lr: 1e-3, weight_decay: 1e-2);
        this.scheduler = optim.lr_scheduler.ReduceLROnPlateau(this.optimizer, patience: 5);
    }

    public async Task TrainOneEpoch(int epoch, CancellationToken stoppingToken) {
        this.model.train();
        var batchIndex = 0;

        await foreach (var (input, target) in this.loader.WithCancellation(stoppingToken)) {
            this.optimizer.zero_grad();

            var output = this.model.forward(input);

            var classificationLoss = this.bce.forward(output[.., 0], target[.., 0]);
            var regressionLoss = this.huber.forward(output[.., 1], target[.., 1]);

            var loss = this.option.Alpha * classificationLoss + (1 - this.option.Alpha) * regressionLoss;
            loss.backward();
            this.optimizer.step();

            if (batchIndex % 10 == 0) {
                this.logger.LogInformation(
                    $"Train Epoch {epoch}, Batch {batchIndex}, Total Loss: {loss.item<float>():F4}");
                this.writer.add_scalar("Train/TotalLoss", loss.item<float>(), epoch * 100 + batchIndex);
                this.writer.add_scalar("Train/ClassificationLoss", classificationLoss.item<float>(),
                    epoch * 100 + batchIndex);
                this.writer.add_scalar("Train/RegressionLoss", regressionLoss.item<float>(), epoch * 100 + batchIndex);
            }

            batchIndex++;
        }
    }

    public async Task<bool> ValidateOneEpoch(int epoch, CancellationToken stoppingToken) {
        this.model.eval();
        var totalLoss = 0.0f;
        var batchIndex = 0;

        await foreach (var (input, target) in this.loader.WithCancellation(stoppingToken)) {
            using var _ = no_grad();
            var output = this.model.forward(input);

            var classificationLoss = this.bce.forward(output[.., 0], target[.., 0]);
            var regressionLoss = this.huber.forward(output[.., 1], target[.., 1]);
            var loss = this.option.Alpha * classificationLoss + (1 - this.option.Alpha) * regressionLoss;

            totalLoss += loss.item<float>();

            if (batchIndex % 10 == 0) {
                this.logger.LogInformation(
                    $"Val Epoch {epoch}, Batch {batchIndex}, Total Loss: {loss.item<float>():F4}");
                this.writer.add_scalar("Validation/TotalLoss", loss.item<float>(), epoch * 100 + batchIndex);
                this.writer.add_scalar("Validation/ClassificationLoss", classificationLoss.item<float>(),
                    epoch * 100 + batchIndex);
                this.writer.add_scalar("Validation/RegressionLoss", regressionLoss.item<float>(),
                    epoch * 100 + batchIndex);
            }

            batchIndex++;
        }

        var avgLoss = totalLoss / batchIndex;
        this.scheduler.step(avgLoss);

        this.logger.LogInformation($"Validation Loss after epoch {epoch}: {avgLoss:F4}");
        this.writer.add_scalar("Validation/Loss", avgLoss, epoch);

        if (avgLoss < this.bestValidationLoss) {
            this.bestValidationLoss = avgLoss;
            this.patienceCounter = 0;
        } else {
            this.patienceCounter++;
            this.logger.LogInformation($"Validation loss did not improve. Patience counter: {this.patienceCounter}/{this.option.Patience}");

            if (this.patienceCounter < this.option.Patience)
                return true;

            this.logger.LogInformation("Early stopping triggered.");
            return false;
        }

        return true;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        for (var epoch = 1; epoch <= this.option.Epochs; epoch++) {
            this.logger.LogInformation($"Starting epoch {epoch}/{this.option.Epochs}");

            await this.TrainOneEpoch(epoch, stoppingToken);
            var improved = await this.ValidateOneEpoch(epoch, stoppingToken);

            if (!improved)
                break;
        }
    }
}
