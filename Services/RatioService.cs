namespace JokeTrader.Services;

using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

internal class RatioService(IBybitRestClient restClient, IDbContextFactory<JokerContext> db,
    IOptions<JokerOption> options, ILogger<RatioService> logger) : BackgroundService {
    public JokerOption Opt => options.Value;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await using var context = await db.CreateDbContextAsync(stoppingToken);
        await this.Prepare(context, context.BTCRatios, "BTCUSDT", this.Opt.HistoryStart, this.Opt.HistoryEnd, stoppingToken);
    }

    public async Task Prepare<T>(JokerContext context, DbSet<T> targetDb, string symbolName, DateTime startTime,
        DateTime endTime, CancellationToken stoppingToken) where T : BasicRatio, new() {

        var symbol = await context.Symbols.FirstAsync(s => s.Name == symbolName, stoppingToken);

        if (await targetDb.AnyAsync(r => r.Timestamp >= endTime, stoppingToken)) {
            logger.LogInformation("{0} Ratios already exist beyond the specified endTime.", symbol.Name);
            return;
        }

        while (startTime < endTime) {
            var ratios = await this.FetchRatios<T>(symbol, startTime, endTime, stoppingToken);

            targetDb.AddRange(ratios);
            await context.SaveChangesAsync(stoppingToken);

            startTime = ratios.MaxBy(r => r.Timestamp)!.Timestamp.AddMinutes(1);
            logger.LogInformation("Fetched {0} Ratios for {1} starting from {2}", ratios.Length, symbol.Name, startTime);
        }

        logger.LogInformation("{0} Ratios prepared", symbol.Name);
    }

    public async Task<T[]> FetchRatios<T>(Symbol symbol, DateTime startTime, DateTime? endTime,
        CancellationToken stoppingToken) where T : BasicRatio, new() {

        var ratioResult = await restClient.V5Api.ExchangeData.GetLongShortRatioAsync(
            this.Opt.Category, symbol.Name, this.Opt.Period, startTime, endTime, 500, stoppingToken);

        if (!ratioResult.Success)
            throw new HttpRequestException(ratioResult.Error?.Message);

        var symbolRatios = ratioResult.Data.Select(r => new T {
            Timestamp = r.Timestamp,
            BuyRatio = (double)r.BuyRatio,
            SellRatio = (double)r.SellRatio,
            Symbol = symbol
        }).ToArray();

        return symbolRatios;
    }
}
