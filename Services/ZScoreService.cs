namespace JokeTrader.Services;

using Microsoft.EntityFrameworkCore;
using Torch;

internal class ZScoreService(IDbContextFactory<JokerContext> db, ILogger<ZScoreService> logger) : BackgroundService {
    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await using var context = await db.CreateDbContextAsync(stoppingToken);
        var symbol = await context.Symbols.FirstAsync(stoppingToken);

        var openPrices = await context.BTCKLines
            .Select(k => k.OpenPrice)
            .ToArrayAsync(stoppingToken);

        var openInterests = await context.BTCInterests
            .Select(i => i.OpenInterest)
            .ToArrayAsync(stoppingToken);

        var priceMean = openPrices.CalculateMean(x => x);
        var priceStd = openPrices.CalculateStd(x => x, priceMean);

        var interestMean = openInterests.CalculateMean(x => x);
        var interestStd = openInterests.CalculateStd(x => x, interestMean);

        var priceNorm = await context.Normalizations
                                .Where(n =>
                                    n.SymbolId == symbol.Name &&
                                    n.Feature == nameof(SeriesFeatures.OpenPrice))
                                .SingleOrDefaultAsync(stoppingToken)
                            ?? context.Normalizations.Add(new() {
                                SymbolId = symbol.Name,
                                Feature = nameof(SeriesFeatures.OpenPrice)
                            }).Entity;

        priceNorm.Mean = priceMean;
        priceNorm.Std = priceStd;

        var interestNorm = await context.Normalizations
                                .Where(n =>
                                    n.SymbolId == symbol.Name &&
                                    n.Feature == nameof(SeriesFeatures.OpenInterest))
                                .SingleOrDefaultAsync(stoppingToken)
                            ?? context.Normalizations.Add(new() {
                                SymbolId = symbol.Name,
                                Feature = nameof(SeriesFeatures.OpenInterest)
                            }).Entity;

        interestNorm.Mean = interestMean;
        interestNorm.Std = interestStd;

        await context.SaveChangesAsync(stoppingToken);
        logger.LogInformation("Price Normalization: {Mean} {Std}", priceMean, priceStd);
        logger.LogInformation("Interest Normalization: {Mean} {Std}", interestMean, interestStd);
    }
}
