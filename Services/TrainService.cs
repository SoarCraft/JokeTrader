namespace JokeTrader.Services;

using Microsoft.Extensions.Options;
using Torch;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

internal class TrainService : BackgroundService {
    public TrainService(JokerDataLoader loader, IOptions<JokerOption> options, ILogger<TrainService> logger) {
        this.loader = loader;
        this.option = options.Value;
        this.logger = logger;

        var featureDim = new SeriesFeatures().ToArray().Length;
        this.model = new(featureDim, this.option.EmbedDim, this.option.NumHeads, this.option.NumLayers);
        this.model.to(this.option.Device);

        this.optimizer = optim.AdamW(this.model.parameters(), weight_decay: 1e-2);
        this.scheduler = optim.lr_scheduler.ReduceLROnPlateau(this.optimizer, patience: 5);
    }

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

    private int trainGlobalStep { get; set; }

    private int valGlobalStep { get; set; }

    public async Task TrainOneEpoch(int epoch, CancellationToken stoppingToken) {
        this.model.train();

        await foreach (var (input, target) in this.loader.WithCancellation(stoppingToken)) {
            this.optimizer.zero_grad();

            var output = this.model.forward(input);

            var classificationLoss = this.bce.forward(output[.., 0], target[.., 0]);
            var regressionLoss = this.huber.forward(output[.., 1], target[.., 1]);

            var loss = this.option.Alpha * classificationLoss + (1 - this.option.Alpha) * regressionLoss;
            loss.backward();
            this.optimizer.step();

            if (this.trainGlobalStep % 10 == 0) {
                this.logger.LogInformation(
                    $"Train Epoch {epoch}, Global Step {this.trainGlobalStep}, Total Loss: {loss.item<float>():F4}");
                this.writer.add_scalar("Train/TotalLoss", loss.item<float>(), this.trainGlobalStep);
                this.writer.add_scalar("Train/ClassificationLoss",
                    classificationLoss.item<float>(), this.trainGlobalStep);
                this.writer.add_scalar("Train/RegressionLoss", regressionLoss.item<float>(), this.trainGlobalStep);
            }

            this.trainGlobalStep++;
            input.Dispose();
            target.Dispose();
        }
    }

    public async Task<bool> ValidateOneEpoch(int epoch, CancellationToken stoppingToken) {
        this.model.eval();
        var totalLoss = 0.0f;
        var step = 0;

        await foreach (var (input, target) in this.loader.WithCancellation(stoppingToken)) {
            using var _ = no_grad();
            var output = this.model.forward(input);

            var classificationLoss = this.bce.forward(output[.., 0], target[.., 0]);
            var regressionLoss = this.huber.forward(output[.., 1], target[.., 1]);
            var loss = this.option.Alpha * classificationLoss + (1 - this.option.Alpha) * regressionLoss;

            totalLoss += loss.item<float>();

            if (this.valGlobalStep % 10 == 0) {
                this.logger.LogInformation(
                    $"Val Epoch {epoch}, Global Step {this.valGlobalStep}, Total Loss: {loss.item<float>():F4}");
                this.writer.add_scalar("Validation/TotalLoss", loss.item<float>(), this.valGlobalStep);
                this.writer.add_scalar("Validation/ClassificationLoss", 
                    classificationLoss.item<float>(), this.valGlobalStep);
                this.writer.add_scalar("Validation/RegressionLoss", regressionLoss.item<float>(), this.valGlobalStep);

                var outputCpu = output.cpu().detach();
                var targetCpu = target.cpu().detach();

                this.writer.add_scalars(
                    "Validation/ClassificationComparison",
                    new Dictionary<string, float> {
                        { "Prediction", outputCpu[.., 0].mean().item<float>() },
                        { "Target", targetCpu[.., 0].mean().item<float>() }
                    },
                    this.valGlobalStep
                );

                this.writer.add_scalars(
                    "Validation/RegressionComparison",
                    new Dictionary<string, float> {
                        { "Prediction", outputCpu[.., 1].mean().item<float>() },
                        { "Target", targetCpu[.., 1].mean().item<float>() }
                    },
                    this.valGlobalStep
                );
            }

            step++;
            this.valGlobalStep++;
            break;
        }

        var avgLoss = totalLoss / step;
        this.scheduler.step(avgLoss);

        this.logger.LogInformation($"Validation Loss after epoch {epoch}: {avgLoss:F4}");
        this.writer.add_scalar("Validation/Loss", avgLoss, epoch);

        if (avgLoss < this.bestValidationLoss) {
            this.bestValidationLoss = avgLoss;
            this.patienceCounter = 0;
        } else {
            this.patienceCounter++;
            this.logger.LogInformation(
                $"Validation loss did not improve. Patience counter: {this.patienceCounter}/{this.option.Patience}");

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
