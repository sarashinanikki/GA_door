using System;
using System.Collections.Generic;
using System.Linq;
using OxyPlot;
using OxyPlot.Series;

public class Individual
{
    public double M { get; set; }
    public double K { get; set; }
    public double D { get; set; }

    public Individual(double m, double k, double d)
    {
        M = m;
        K = k;
        D = d;
    }
}

public class GeneticAlgorithm
{
    private static Random random = new Random();

    public static List<Individual> GenerateInitialPopulation(int populationSize)
    {
        List<Individual> population = new List<Individual>();
        for (int i = 0; i < populationSize; i++)
        {
            double m = random.NextDouble() * 20 + 1; // 0 to 50 kg
            double k = random.NextDouble() * 10 + 10; // 0 to 1000 N/m
            double d = random.NextDouble() * 50 + 10; // 0 to 100 Ns/m
            population.Add(new Individual(m, k, d));
        }
        return population;
    }

    public static Individual Crossover(Individual parent1, Individual parent2)
    {
        double m = (parent1.M + parent2.M) / 2;
        double k = (parent1.K + parent2.K) / 2;
        double d = (parent1.D + parent2.D) / 2;
        return new Individual(m, k, d);
    }

    public static Individual Mutate(Individual individual, double mutationRate)
    {
        if (random.NextDouble() < mutationRate)
        {
            individual.M += random.NextDouble() - 0.5;
            individual.K += random.NextDouble() * 10 - 5;
            individual.D += random.NextDouble() - 0.5;
        }
        return individual;
    }

    public static Individual Select(List<Individual> population)
    {
        double totalFitness = population.Sum(ind => EvaluateFitness(ind));
        double randomValue = random.NextDouble() * totalFitness;
        Console.WriteLine($"totalFitness: {totalFitness}, randomValue: {randomValue}");
        double cumulativeFitness = 0;
        foreach (var individual in population)
        {
            cumulativeFitness += EvaluateFitness(individual);
            if (cumulativeFitness >= randomValue)
            {
                Console.WriteLine(EvaluateFitness(individual));
                return individual;
            }
        }
        return population.Last();
    }

    public static double EvaluateFitness(Individual individual)
    {
        double F = 50; // ドアを開ける力
        double tEnd = 20; // シミュレーションの終了時間
        double dt = 0.1; // 時間刻み
        double[] positions = RungeKutta.Simulate(individual.M, individual.K, individual.D, F, 0, 0, tEnd, dt);

        // 閉じる時間を計算
        double closeTime = Int32.MaxValue;
        for (int i = 10; i < positions.Length; i++)
        {
            if (positions[i] <= 0.01)
            {
                closeTime = i * dt;
                Console.WriteLine($"closeTime: {closeTime}");            
                break;
            }
        }

        if (closeTime == Int32.MaxValue)
        {
            return 0; // ドアが閉じない場合は適応度を低くする
        }

        var minimum_val = positions.Min();
        if (minimum_val < -0.01 || minimum_val > 0.01)
        {
            return 0; // ドアが開きすぎている場合も適応度を低くする
        }

        if (positions.Last() > 0.01 || positions.Last() < -0.01)
        {
            return 0; // ドアが30秒後にも閉じていなかったり、閉じすぎている場合も適応度を低くする
        }

        // 静かさを評価 (振動の振幅が小さいほど適応度が高い)
        double smoothness = 0;
        for (int i = 1; i < positions.Length; i++)
        {
            if (positions[i] > positions[i - 1])
            {
                continue;
            }
            smoothness += positions[i-1] - positions[i];
        }

        // 適応度を計算
        // 静かさが低く、閉じる時間が長いほど適応度が高い
        return (1 / smoothness) + closeTime;
    }
}

public class RungeKutta
{
    public static double[] Simulate(double m, double k, double d, double F, double x0, double v0, double tEnd, double dt)
    {
        int steps = (int)(tEnd / dt);
        double[] positions = new double[steps];
        double x = x0;
        double v = v0;

        for (int i = 0; i < steps; i++)
        {
            double t = i * dt;
            positions[i] = x;

            // 4次のRunge-Kutta法の係数を計算
            double k1_v = dt * (F - k * x - d * v) / m;
            double k1_x = dt * v;

            double k2_v = dt * (F - k * (x + 0.5 * k1_x) - d * (v + 0.5 * k1_v)) / m;
            double k2_x = dt * (v + 0.5 * k1_v);

            double k3_v = dt * (F - k * (x + 0.5 * k2_x) - d * (v + 0.5 * k2_v)) / m;
            double k3_x = dt * (v + 0.5 * k2_v);

            double k4_v = dt * (F - k * (x + k3_x) - d * (v + k3_v)) / m;
            double k4_x = dt * (v + k3_v);

            // 位置と速度の更新
            x += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6;
            v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6;

            if (i >= 5) {
                F = 0;
            }
        }

        return positions;
    }
}


public class Program
{
    public static void Main()
    {
        int populationSize = 100;
        int generations = 50;
        double mutationRate = 0.05;

        var population = GeneticAlgorithm.GenerateInitialPopulation(populationSize);
        for (int i = 0; i < generations; i++)
        {
            var newPopulation = new List<Individual>();
            for (int j = 0; j < populationSize / 2; j++)
            {
                var parent1 = GeneticAlgorithm.Select(population);
                var parent2 = GeneticAlgorithm.Select(population);
                var child1 = GeneticAlgorithm.Crossover(parent1, parent2);
                var child2 = GeneticAlgorithm.Crossover(parent2, parent1);
                newPopulation.Add(GeneticAlgorithm.Mutate(child1, mutationRate));
                newPopulation.Add(GeneticAlgorithm.Mutate(child2, mutationRate));
            }
            population = newPopulation;
        }

        // 降順でソートして最も適応度の高い個体を取得
        var bestIndividual = population.OrderByDescending(ind => GeneticAlgorithm.EvaluateFitness(ind)).First();

        // var bestIndividual = population.OrderBy(ind => GeneticAlgorithm.EvaluateFitness(ind)).First();
        double[] positions = RungeKutta.Simulate(bestIndividual.M, bestIndividual.K, bestIndividual.D, 50, 0, 0, 20, 0.1);



        // bestIndividualのスコアが知りたい
        Console.WriteLine(GeneticAlgorithm.EvaluateFitness(bestIndividual));
        Console.WriteLine($"M: {bestIndividual.M}, K: {bestIndividual.K}, D: {bestIndividual.D}");

        // ドアの開閉シミュレーションの数値をCSVで出力
        using (var writer = new System.IO.StreamWriter("positions.csv"))
        {
            for (int i = 0; i < positions.Length; i++)
            {
                double time = i * 0.1;
                writer.WriteLine($"{time.ToString("F1")},{positions[i]}");
            }
        }
    }
}
