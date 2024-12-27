namespace JokeTrader.Services;

using System.Threading.Tasks;
using Torch;

internal class Orchestration(FundRateService fundRate, InterestService interest, 
    KLineService kLine, RatioService ratio, SymbolService symbol,
    JokerDataLoader loader) : BackgroundService {

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await symbol.StartAsync(stoppingToken);
        await symbol.ExecuteTask!;

        await fundRate.StartAsync(stoppingToken);
        await interest.StartAsync(stoppingToken);
        await kLine.StartAsync(stoppingToken);
        await ratio.StartAsync(stoppingToken);
    }
}
