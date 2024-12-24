namespace JokeTrader;

using Bybit.Net.Enums;
using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

internal class KLineService(IBybitRestClient restClient, JokerContext context, ILogger<KLineService> logger)
    : BackgroundService {
    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        var startTime = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var endTime = new DateTime(2024, 12, 1, 0, 0, 0, DateTimeKind.Utc);

        await this.PrepareBTCUSDT(startTime, endTime, stoppingToken);
    }

    public async Task<Symbol> PrepareSymbol(string symbolName, CancellationToken stoppingToken) {
        var symbolResult = await restClient.V5Api.ExchangeData.GetSpotSymbolsAsync(symbolName, stoppingToken);
        var symbol = symbolResult.Data.List.First();

        var dbSymbol = await context.Symbols.FirstOrDefaultAsync(s => s.Name == symbol.Name, stoppingToken)
                       ?? context.Symbols.Add(new() {
                           Name = symbol.Name
                       }).Entity;

        dbSymbol.MaxOrderValue = (double)symbol.LotSizeFilter!.MaxOrderValue;
        dbSymbol.MinOrderValue = (double)symbol.LotSizeFilter!.MinOrderValue;
        dbSymbol.LastUpdated = DateTime.UtcNow;

        await context.SaveChangesAsync(stoppingToken);
        logger.LogInformation("Symbol {0} prepared", dbSymbol.Name);
        return dbSymbol;
    }

    public async Task PrepareBTCUSDT(DateTime startTime, DateTime endTime, CancellationToken stoppingToken) {
        var dbSymbol = await this.PrepareSymbol("BTCUSDT", stoppingToken);

        if (await context.BTCKLines.AnyAsync(k => k.StartTime >= endTime, stoppingToken)) {
            logger.LogInformation("BTCUSDT KLines already exist beyond the specified endTime.");
            return;
        }

        while (startTime < endTime) {
            var kLines = await this.FetchKLines<BTCKLine>(
                dbSymbol, KlineInterval.OneMinute, startTime, endTime, stoppingToken);

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
