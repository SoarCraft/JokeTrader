namespace JokeTrader;

using Microsoft.EntityFrameworkCore;

public class JokerContext(DbContextOptions<JokerContext> options) : DbContext(options) {
    public DbSet<Symbol> Symbols { get; set; }

    public DbSet<BTCKLine> BTCKLines { get; set; }

    public DbSet<BTCRatio> BTCRatios { get; set; }

    public DbSet<BTCFundRate> BTCFundRates { get; set; }

    public DbSet<BTCInterest> BTCInterests { get; set; }
}
