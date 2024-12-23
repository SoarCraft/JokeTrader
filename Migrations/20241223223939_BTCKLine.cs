using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace JokeTrader.Migrations
{
    /// <inheritdoc />
    public partial class BTCKLine : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "Symbols",
                columns: table => new
                {
                    Name = table.Column<string>(type: "nvarchar(450)", nullable: false),
                    MaxOrderValue = table.Column<double>(type: "float", nullable: false),
                    MinOrderValue = table.Column<double>(type: "float", nullable: false),
                    LastUpdated = table.Column<DateTime>(type: "datetime2", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Symbols", x => x.Name);
                });

            migrationBuilder.CreateTable(
                name: "BTCKLines",
                columns: table => new
                {
                    StartTime = table.Column<DateTime>(type: "datetime2", nullable: false),
                    OpenPrice = table.Column<double>(type: "float", nullable: false),
                    SymbolName = table.Column<string>(type: "nvarchar(450)", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_BTCKLines", x => x.StartTime);
                    table.ForeignKey(
                        name: "FK_BTCKLines_Symbols_SymbolName",
                        column: x => x.SymbolName,
                        principalTable: "Symbols",
                        principalColumn: "Name",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_BTCKLines_SymbolName",
                table: "BTCKLines",
                column: "SymbolName");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "BTCKLines");

            migrationBuilder.DropTable(
                name: "Symbols");
        }
    }
}
