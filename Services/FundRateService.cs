namespace JokeTrader.Services;

using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

internal class FundRateService(IBybitRestClient restClient, JokerContext context,
    IOptions<JokerOption> options, ILogger<FundRateService> logger) : BackgroundService {
    public JokerOption Opt => options.Value;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await this.PrepareBTCUSDT(this.Opt.HistoryStart, this.Opt.HistoryEnd, stoppingToken);
    }

    public async Task PrepareBTCUSDT(DateTime startTime, DateTime endTime, CancellationToken stoppingToken) {
        var symbol = await context.Symbols.FirstAsync(s => s.Name == "BTCUSDT", stoppingToken);

        var rate = await restClient.V5Api.ExchangeData.GetFundingRateHistoryAsync(
            this.Opt.Category, symbol.Name, startTime, endTime, ct: stoppingToken);
    }
}
