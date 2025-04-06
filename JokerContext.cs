namespace JokeTrader;

using Microsoft.EntityFrameworkCore;

public class JokerContext(DbContextOptions<JokerContext> options) : DbContext(options) {
    public DbSet<Symbol> Symbols { get; set; }

    public DbSet<KLine> KLines { get; set; }

    public DbSet<Ratio> Ratios { get; set; }

    public DbSet<FundRate> FundRates { get; set; }

    public DbSet<Interest> Interests { get; set; }
}
