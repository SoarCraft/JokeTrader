namespace JokeTrader.Services;

using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

internal class SymbolService(
    IBybitRestClient restClient, JokerContext context, IOptions<JokerOption> options,
    ILogger<SymbolService> logger) : BackgroundService {

    public async Task Prepare() {
        var opt = options.Value;

        foreach (var symbolName in opt.Symbols) {
            var symbolResult = await restClient.V5Api.ExchangeData.GetSpotSymbolsAsync(symbolName);
            var symbol = symbolResult.Data.List.First();

            var dbSymbol = await context.Symbols.FirstOrDefaultAsync(s => s.Name == symbol.Name)
                           ?? context.Symbols.Add(new() {
                               Name = symbol.Name
                           }).Entity;

            dbSymbol.MaxOrderValue = (double)symbol.LotSizeFilter!.MaxOrderValue;
            dbSymbol.MinOrderValue = (double)symbol.LotSizeFilter!.MinOrderValue;
            dbSymbol.LastUpdated = DateTime.UtcNow;

            logger.LogInformation("Symbol {0} prepared", dbSymbol.Name);
        }

        await context.SaveChangesAsync();
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        // TODO
    }
}
