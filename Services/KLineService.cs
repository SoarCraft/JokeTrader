namespace JokeTrader.Services;

using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

internal class KLineService(IBybitRestClient restClient, IDbContextFactory<JokerContext> db, 
    IOptions<JokerOption> options, ILogger<KLineService> logger) : BackgroundService {
    public JokerOption Opt => options.Value;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await using var context = await db.CreateDbContextAsync(stoppingToken);
        await this.Prepare(context, context.KLines, this.Opt.Symbol, this.Opt.HistoryStart, this.Opt.HistoryEnd, stoppingToken);
    }

    public async Task Prepare<T>(JokerContext context, DbSet<T> targetDb, string symbolName, DateTime startTime,
        DateTime endTime, CancellationToken stoppingToken) where T : KLine, new() {

        var symbol = await context.Symbols.FirstAsync(s => s.Name == symbolName, stoppingToken);

        if (await targetDb.AnyAsync(k => k.StartTime >= endTime, stoppingToken)) {
            logger.LogInformation("{0} KLines already exist beyond the specified endTime.", symbol.Name);
            return;
        }

        while (endTime > startTime) {
            var kLines = await this.FetchKLines<T>(symbol, startTime, endTime, stoppingToken);

            targetDb.AddRange(kLines);
            await context.SaveChangesAsync(stoppingToken);

            logger.LogInformation("Fetched {0} KLines for {1} up to {2}", kLines.Length, symbol.Name, endTime);
            endTime = kLines.MinBy(k => k.StartTime)!.StartTime.AddMinutes(-1);
        }

        logger.LogInformation("{0} KLines prepared", symbol.Name);
    }

    public async Task<T[]> FetchKLines<T>(Symbol symbol, DateTime startTime, DateTime? endTime,
        CancellationToken stoppingToken) where T : KLine, new() {

        var klineResult = await restClient.V5Api.ExchangeData.GetKlinesAsync(
            this.Opt.Category, symbol.Name, this.Opt.KlineInterval, startTime, endTime, 1000, stoppingToken);

        if (!klineResult.Success)
            throw new HttpRequestException(klineResult.Error?.Message);

        var symbolLines = klineResult.Data.List.Select(k => new T {
            StartTime = k.StartTime,
            OpenPrice = (double)k.OpenPrice,
            HighPrice = (double)k.HighPrice,
            LowPrice = (double)k.LowPrice,
            ClosePrice = (double)k.ClosePrice,
            Volume = (double)k.Volume,
            Symbol = symbol
        }).ToArray();

        return symbolLines;
    }
}
