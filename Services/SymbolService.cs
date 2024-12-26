namespace JokeTrader.Services;

using Bybit.Net.Interfaces.Clients;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

internal class SymbolService(
    IBybitRestClient restClient, IDbContextFactory<JokerContext> db, IOptions<JokerOption> options,
    ILogger<SymbolService> logger) : BackgroundService {
    public JokerOption Opt => options.Value;

    public async Task Fetch(JokerContext context) {
        foreach (var symbolName in this.Opt.Symbols) {
            var symbolResult = await restClient.V5Api.ExchangeData.GetLinearInverseSymbolsAsync(this.Opt.Category, symbolName);
            var symbol = symbolResult.Data.List.First();

            var dbSymbol = await context.Symbols.FirstOrDefaultAsync(s => s.Name == symbol.Name)
                           ?? context.Symbols.Add(new() {
                               Name = symbol.Name
                           }).Entity;

            dbSymbol.MaxPrice = (double)symbol.PriceFilter!.MaxPrice;
            dbSymbol.MinPrice = (double)symbol.PriceFilter!.MinPrice;
            dbSymbol.MaxLeverage = (double)symbol.LeverageFilter!.MaxLeverage;
            dbSymbol.MinLeverage = (double)symbol.LeverageFilter!.MinLeverage;
            dbSymbol.LastUpdated = DateTime.UtcNow;

            logger.LogInformation("Symbol {0} prepared", dbSymbol.Name);
        }

        await context.SaveChangesAsync();
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await using var context = await db.CreateDbContextAsync(stoppingToken);
        await this.Fetch(context);
    }
}
