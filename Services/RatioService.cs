﻿namespace JokeTrader.Services;

using System.Threading.Tasks;
using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

internal class RatioService(IBybitRestClient restClient, JokerContext context, 
    IOptions<JokerOption> options, ILogger<RatioService> logger) : BackgroundService {
    public JokerOption Opt => options.Value;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await this.PrepareBTCUSDT(this.Opt.HistoryStart, this.Opt.HistoryEnd, stoppingToken);
    }

    public async Task PrepareBTCUSDT(DateTime startTime, DateTime endTime, CancellationToken stoppingToken) {
        var symbol = await context.Symbols.FirstAsync(s => s.Name == "BTCUSDT", stoppingToken);

        var ratio = await restClient.V5Api.ExchangeData.GetLongShortRatioAsync(
            this.Opt.Category, symbol.Name, this.Opt.Period, startTime, endTime, 500, stoppingToken);
    }
}
