namespace JokeTrader;

public static class Extensions {
    public static double CalculateMean<T>(this IEnumerable<T> source, Func<T, double> selector) {
        var array = source as T[] ?? source.ToArray();

        var sum = array.Select(selector).Sum();
        var count = array.Length;

        return count > 0 ? sum / count : 0;
    }

    public static double CalculateStd<T>(this IEnumerable<T> source, Func<T, double> selector, double mean) {
        var array = source as T[] ?? source.ToArray();

        var count = array.Length;
        if (count <= 1)
            return 0;

        var sum = array.Select(data => Math.Pow(selector(data) - mean, 2)).Sum();
        return Math.Sqrt(sum / (count - 1));
    }
}
