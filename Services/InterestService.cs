namespace JokeTrader.Services;

using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

internal class InterestService(IBybitRestClient restClient, JokerContext context,
    IOptions<JokerOption> options, ILogger<InterestService> logger) : BackgroundService {
    public JokerOption Opt => options.Value;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await this.Prepare(context.BTCInterests, "BTCUSDT", this.Opt.HistoryStart, this.Opt.HistoryEnd, stoppingToken);
    }

    public async Task Prepare<T>(DbSet<T> targetDb, string symbolName, DateTime startTime,
        DateTime endTime, CancellationToken stoppingToken) where T : BasicInterest, new() {

        var symbol = await context.Symbols.FirstAsync(s => s.Name == symbolName, stoppingToken);

        if (await targetDb.AnyAsync(r => r.Timestamp >= endTime, stoppingToken)) {
            logger.LogInformation("{0} Open Interests already exist beyond the specified endTime.", symbol.Name);
            return;
        }

        while (startTime < endTime) {
            var interests = await this.FetchInterests<T>(symbol, startTime, endTime, stoppingToken);

            targetDb.AddRange(interests);
            await context.SaveChangesAsync(stoppingToken);

            startTime = interests.MaxBy(r => r.Timestamp)!.Timestamp.AddMinutes(1);
            logger.LogInformation("Fetched {0} Open Interests for {1} starting from {2}", interests.Length, symbol.Name, startTime);
        }

        logger.LogInformation("{0} Open Interests prepared", symbol.Name);
    }

    public async Task<T[]> FetchInterests<T>(Symbol symbol, DateTime startTime, DateTime? endTime,
        CancellationToken stoppingToken) where T : BasicInterest, new() {

        var interestResult = await restClient.V5Api.ExchangeData.GetOpenInterestAsync(
            this.Opt.Category, symbol.Name, this.Opt.InterestInterval, startTime, endTime, ct: stoppingToken);

        if (!interestResult.Success)
            throw new HttpRequestException(interestResult.Error?.Message);

        var interests = interestResult.Data.List.Select(r => new T {
            Timestamp = r.Timestamp,
            OpenInterest = r.OpenInterest,
            Symbol = symbol
        }).ToArray();

        return interests;
    }
}
