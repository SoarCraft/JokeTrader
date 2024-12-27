namespace JokeTrader;

using Microsoft.EntityFrameworkCore;

public class JokerContext(DbContextOptions<JokerContext> options) : DbContext(options) {
    public DbSet<Symbol> Symbols { get; set; }

    public DbSet<Normalization> Normalizations { get; set; }

    public DbSet<BTCKLine> BTCKLines { get; set; }

    public DbSet<BTCRatio> BTCRatios { get; set; }

    public DbSet<BTCFundRate> BTCFundRates { get; set; }

    public DbSet<BTCInterest> BTCInterests { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder) {
        base.OnModelCreating(modelBuilder);

        modelBuilder.Entity<Normalization>()
            .HasKey(n => new { n.SymbolId, n.Feature });

        modelBuilder.Entity<Normalization>()
            .HasOne(n => n.Symbol)
            .WithMany(s => s.Normalizations)
            .HasForeignKey(n => n.SymbolId);
    }
}
