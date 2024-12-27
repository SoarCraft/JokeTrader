namespace JokeTrader.Services;

using Microsoft.EntityFrameworkCore;

internal class ZScoreService(IDbContextFactory<JokerContext> db, ILogger<ZScoreService> logger) : BackgroundService {
    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {

    }
}
