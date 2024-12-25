﻿namespace JokeTrader.Services;

using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

internal class FundRateService(IBybitRestClient restClient, JokerContext context,
    IOptions<JokerOption> options, ILogger<FundRateService> logger) : BackgroundService {
    public JokerOption Opt => options.Value;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await this.Prepare(context.BTCFundRates, "BTCUSDT", this.Opt.HistoryStart, this.Opt.HistoryEnd, stoppingToken);
    }

    public async Task Prepare<T>(DbSet<T> targetDb, string symbolName, DateTime startTime,
        DateTime endTime, CancellationToken stoppingToken) where T : BasicFundRate, new() {

        var symbol = await context.Symbols.FirstAsync(s => s.Name == symbolName, stoppingToken);

        if (await targetDb.AnyAsync(r => r.Timestamp >= endTime, stoppingToken)) {
            logger.LogInformation("{0} Funding Rates already exist beyond the specified endTime.", symbol.Name);
            return;
        }

        while (startTime < endTime) {
            var rates = await this.FetchFundRates<T>(symbol, startTime, endTime, stoppingToken);

            targetDb.AddRange(rates);
            await context.SaveChangesAsync(stoppingToken);

            startTime = rates.MaxBy(r => r.Timestamp)!.Timestamp.AddMinutes(symbol.FundingInterval);
            logger.LogInformation("Fetched {0} Funding Rates for {1} starting from {2}", rates.Length, symbol.Name, startTime);
        }

        logger.LogInformation("{0} Funding Rates prepared", symbol.Name);
    }

    public async Task<T[]> FetchFundRates<T>(Symbol symbol, DateTime startTime, DateTime? endTime,
        CancellationToken stoppingToken) where T : BasicFundRate, new() {

        var rateResult = await restClient.V5Api.ExchangeData.GetFundingRateHistoryAsync(
            this.Opt.Category, symbol.Name, startTime, endTime, ct: stoppingToken);

        if (!rateResult.Success)
            throw new HttpRequestException(rateResult.Error?.Message);

        var fundRates = rateResult.Data.List.Select(r => new T {
            Timestamp = r.Timestamp,
            FundingRate = (double)r.FundingRate,
            Symbol = symbol
        }).ToArray();

        return fundRates;
    }
}
