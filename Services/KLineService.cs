namespace JokeTrader.Services;

using Bybit.Net.Enums;
using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

internal class KLineService(IBybitRestClient restClient, JokerContext context, 
    IOptions<JokerOption> options, ILogger<KLineService> logger) : BackgroundService {
    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        var opt = options.Value;

        await this.PrepareBTCUSDT(opt.HistoryStart, opt.HistoryEnd, stoppingToken);
    }

    public async Task PrepareBTCUSDT(DateTime startTime, DateTime endTime, CancellationToken stoppingToken) {
        var symbol = await context.Symbols.FirstAsync(s => s.Name == "BTCUSDT", stoppingToken);

        if (await context.BTCKLines.AnyAsync(k => k.StartTime >= endTime, stoppingToken)) {
            logger.LogInformation("BTCUSDT KLines already exist beyond the specified endTime.");
            return;
        }

        while (startTime < endTime) {
            var kLines = await this.FetchKLines<BTCKLine>(
                symbol, KlineInterval.OneMinute, startTime, endTime, stoppingToken);

            context.BTCKLines.AddRange(kLines);
            await context.SaveChangesAsync(stoppingToken);

            startTime = kLines.MaxBy(k => k.StartTime)!.StartTime.AddMinutes(1); 
            logger.LogInformation("Fetched {0} KLines starting from {1}", kLines.Length, startTime);
        }

        logger.LogInformation("BTCUSDT KLines prepared");
    }

    public async Task<T[]> FetchKLines<T>(Symbol symbol, KlineInterval interval, DateTime startTime, DateTime? endTime,
        CancellationToken stoppingToken) where T : BasicKLine, new() {

        var klineResult = await restClient.V5Api.ExchangeData.GetKlinesAsync(
            Category.Spot, symbol.Name, interval, startTime, endTime, 1000, stoppingToken);

        if (!klineResult.Success)
            throw new HttpRequestException(klineResult.Error?.Message);

        var symbolLines = klineResult.Data.List.Select(k => new T {
            StartTime = k.StartTime,
            OpenPrice = (double)k.OpenPrice,
            Symbol = symbol
        }).ToArray();

        return symbolLines;
    }
}
