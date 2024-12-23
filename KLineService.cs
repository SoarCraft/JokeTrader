namespace JokeTrader;

using Bybit.Net.Enums;
using Bybit.Net.Interfaces.Clients;
using Microsoft.Extensions.Hosting;

internal class KLineService(IBybitRestClient restClient) : BackgroundService {

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        var symbolResult = await restClient.V5Api.ExchangeData.GetSpotSymbolsAsync("BTCUSDT", stoppingToken);
        var symbol = symbolResult.Data.List.First();

        var startTime = DateTime.UtcNow.AddDays(-7);
        var endTime = DateTime.UtcNow;

        var klineResult = await restClient.V5Api.ExchangeData.GetKlinesAsync(
            Category.Spot, symbol.Name, KlineInterval.OneDay, startTime, endTime, ct: stoppingToken);

        if (klineResult.Success) {
            var kLines = klineResult.Data.List;

        }
    }
}
