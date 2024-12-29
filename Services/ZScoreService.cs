namespace JokeTrader.Services;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Torch;

internal class ZScoreService(
    IDbContextFactory<JokerContext> db,
    IOptions<JokerOption> options,
    ILogger<ZScoreService> logger) : BackgroundService {
    public JokerOption Opt => options.Value;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        await using var context = await db.CreateDbContextAsync(stoppingToken);

        var priceNorm = await context.Normalizations
            .Where(n =>
                n.SymbolId == this.Opt.Symbol &&
                n.Feature == nameof(SeriesFeatures.OpenPrice))
            .SingleOrDefaultAsync(stoppingToken);

        if (priceNorm == null) {
            var openPrices = await context.BTCKLines
                .Select(k => k.OpenPrice)
                .ToArrayAsync(stoppingToken);

            var priceMean = openPrices.CalculateMean(x => x);
            var priceStd = openPrices.CalculateStd(x => x, priceMean);

            context.Normalizations.Add(new() {
                SymbolId = this.Opt.Symbol,
                Feature = nameof(SeriesFeatures.OpenPrice),
                Mean = priceMean,
                Std = priceStd
            });

            logger.LogInformation("Price Normalization: {Mean} {Std}", priceMean, priceStd);
        }

        var interestNorm = await context.Normalizations
            .Where(n =>
                n.SymbolId == this.Opt.Symbol &&
                n.Feature == nameof(SeriesFeatures.OpenInterest))
            .SingleOrDefaultAsync(stoppingToken);

        if (interestNorm == null) {
            var openInterests = await context.BTCInterests
                .Select(i => i.OpenInterest)
                .ToArrayAsync(stoppingToken);

            var interestMean = openInterests.CalculateMean(x => x);
            var interestStd = openInterests.CalculateStd(x => x, interestMean);

            context.Normalizations.Add(new() {
                SymbolId = this.Opt.Symbol,
                Feature = nameof(SeriesFeatures.OpenInterest),
                Mean = interestMean,
                Std = interestStd
            });

            logger.LogInformation("Interest Normalization: {Mean} {Std}", interestMean, interestStd);
        }

        await context.SaveChangesAsync(stoppingToken);
    }
}
