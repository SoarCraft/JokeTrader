namespace JokeTrader.Services;

using System.Threading.Tasks;

internal class Orchestration(FundRateService fundRate, InterestService interest, 
    KLineService kLine, RatioService ratio, SymbolService symbol,
    ZScoreService zScore, TrainService train) : BackgroundService {

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await symbol.StartAsync(stoppingToken);
        await symbol.ExecuteTask!;

        await fundRate.StartAsync(stoppingToken);
        await interest.StartAsync(stoppingToken);
        await kLine.StartAsync(stoppingToken);
        await ratio.StartAsync(stoppingToken);

        Task.WaitAll([
            fundRate.ExecuteTask!,
            interest.ExecuteTask!,
            kLine.ExecuteTask!,
            ratio.ExecuteTask!
        ]);

        await zScore.StartAsync(stoppingToken);
        await zScore.ExecuteTask!;

        await train.StartAsync(stoppingToken);
    }
}
