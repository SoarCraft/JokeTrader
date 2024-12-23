﻿// <auto-generated />
using System;
using JokeTrader;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Migrations;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

#nullable disable

namespace JokeTrader.Migrations
{
    [DbContext(typeof(JokerContext))]
    [Migration("20241223223939_BTCKLine")]
    partial class BTCKLine
    {
        /// <inheritdoc />
        protected override void BuildTargetModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder
                .HasAnnotation("ProductVersion", "9.0.0")
                .HasAnnotation("Relational:MaxIdentifierLength", 128);

            SqlServerModelBuilderExtensions.UseIdentityColumns(modelBuilder);

            modelBuilder.Entity("JokeTrader.BTCKLine", b =>
                {
                    b.Property<DateTime>("StartTime")
                        .HasColumnType("datetime2");

                    b.Property<double>("OpenPrice")
                        .HasColumnType("float");

                    b.Property<string>("SymbolName")
                        .IsRequired()
                        .HasColumnType("nvarchar(450)");

                    b.HasKey("StartTime");

                    b.HasIndex("SymbolName");

                    b.ToTable("BTCKLines");
                });

            modelBuilder.Entity("JokeTrader.Symbol", b =>
                {
                    b.Property<string>("Name")
                        .HasColumnType("nvarchar(450)");

                    b.Property<DateTime>("LastUpdated")
                        .HasColumnType("datetime2");

                    b.Property<double>("MaxOrderValue")
                        .HasColumnType("float");

                    b.Property<double>("MinOrderValue")
                        .HasColumnType("float");

                    b.HasKey("Name");

                    b.ToTable("Symbols");
                });

            modelBuilder.Entity("JokeTrader.BTCKLine", b =>
                {
                    b.HasOne("JokeTrader.Symbol", "Symbol")
                        .WithMany("BTCKLines")
                        .HasForeignKey("SymbolName")
                        .OnDelete(DeleteBehavior.Cascade)
                        .IsRequired();

                    b.Navigation("Symbol");
                });

            modelBuilder.Entity("JokeTrader.Symbol", b =>
                {
                    b.Navigation("BTCKLines");
                });
#pragma warning restore 612, 618
        }
    }
}