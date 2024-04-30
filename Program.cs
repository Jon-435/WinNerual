using System;
using System.Collections.Generic;
using System.IO;

public class NeuralNetwork
{
    private readonly int inputNodes;
    private readonly int hiddenNodes;
    private readonly int outputNodes;
    private readonly double[,] weightsInputHidden;
    private readonly double[,] weightsHiddenOutput;
    private readonly double[] hiddenBiases;
    private readonly double[] outputBiases;
    public double learningRate;
    public int maxEpochs;
    public double targetError;
    private readonly Random rand = new Random();
    private double initialLearningRate;
    private double currentLearningRate;

    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double initialLearningRate)
    {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        weightsInputHidden = new double[inputNodes, hiddenNodes];
        weightsHiddenOutput = new double[hiddenNodes, outputNodes];
        hiddenBiases = new double[hiddenNodes];
        outputBiases = new double[outputNodes];
        this.initialLearningRate = initialLearningRate;
        this.currentLearningRate = initialLearningRate;
        InitializeWeights();
        InitializeBiases();
    }

    private void InitializeWeights()
    {
        for (int i = 0; i < inputNodes; i++)
        {
            for (int j = 0; j < hiddenNodes; j++)
            {
                weightsInputHidden[i, j] = rand.NextDouble() * 2 - 1;
            }
        }
        for (int i = 0; i < hiddenNodes; i++)
        {
            for (int j = 0; j < outputNodes; j++)
            {
                weightsHiddenOutput[i, j] = rand.NextDouble() * 2 - 1;
            }
        }
    }

    private void InitializeBiases()
    {
        for (int i = 0; i < hiddenNodes; i++)
        {
            hiddenBiases[i] = rand.NextDouble() * 2 - 1;
        }
        for (int i = 0; i < outputNodes; i++)
        {
            outputBiases[i] = rand.NextDouble() * 2 - 1;
        }
    }

    private double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    private double[] ForwardPropagation(double[] inputs)
    {
        double[] hiddenOutputs = new double[hiddenNodes];
        double[] finalOutputs = new double[outputNodes];
        for (int i = 0; i < hiddenNodes; i++)
        {
            double sum = 0;
            for (int j = 0; j < inputNodes; j++)
            {
                sum += inputs[j] * weightsInputHidden[j, i];
            }
            sum += hiddenBiases[i];
            hiddenOutputs[i] = Sigmoid(sum);
        }
        for (int i = 0; i < outputNodes; i++)
        {
            double sum = 0;
            for (int j = 0; j < hiddenNodes; j++)
            {
                sum += hiddenOutputs[j] * weightsHiddenOutput[j, i];
            }
            sum += outputBiases[i];
            finalOutputs[i] = sum;
        }
        return finalOutputs;
    }

    private double BackPropagation(double[] inputs, double[] targets)
    {
        double[] hiddenOutputs = new double[hiddenNodes];
        double[] finalOutputs = ForwardPropagation(inputs);
        double[] outputErrors = new double[outputNodes];
        double[] outputDeltas = new double[outputNodes];
        for (int i = 0; i < outputNodes; i++)
        {
            outputErrors[i] = targets[i] - finalOutputs[i];
            outputDeltas[i] = outputErrors[i];
        }
        for (int i = 0; i < hiddenNodes; i++)
        {
            for (int j = 0; j < outputNodes; j++)
            {
                weightsHiddenOutput[i, j] += learningRate * outputDeltas[j] * hiddenOutputs[i];
            }
        }
        for (int i = 0; i < outputNodes; i++)
        {
            outputBiases[i] += learningRate * outputDeltas[i];
        }
        double[] hiddenErrors = new double[hiddenNodes];
        double[] hiddenDeltas = new double[hiddenNodes];
        for (int i = 0; i < hiddenNodes; i++)
        {
            double error = 0;
            for (int j = 0; j < outputNodes; j++)
            {
                error += outputDeltas[j] * weightsHiddenOutput[i, j];
            }
            hiddenErrors[i] = error;
            hiddenDeltas[i] = hiddenOutputs[i] * (1 - hiddenOutputs[i]) * error;
        }
        for (int i = 0; i < inputNodes; i++)
        {
            for (int j = 0; j < hiddenNodes; j++)
            {
                weightsInputHidden[i, j] += learningRate * hiddenDeltas[j] * inputs[i];
            }
        }
        for (int i = 0; i < hiddenNodes; i++)
        {
            hiddenBiases[i] += learningRate * hiddenDeltas[i];
        }
        double totalError = 0;
        for (int i = 0; i < outputNodes; i++)
        {
            totalError += Math.Pow(outputErrors[i], 2);
        }
        totalError /= outputNodes;
        totalError = Math.Sqrt(totalError);
        return totalError;
    }

    private void AdjustLearningRate(int epoch)
    {
        double decayRate = 0.95;
        currentLearningRate = initialLearningRate * Math.Pow(decayRate, epoch);
    }

    public void Train(double[][] trainingInputs, double[][] trainingOutputs)
    {
        int epochs = 0;
        double totalError = double.MaxValue;
        while (epochs < maxEpochs && totalError > targetError)
        {
            totalError = 0;
            for (int i = 0; i < trainingInputs.Length; i++)
            {
                totalError += BackPropagation(trainingInputs[i], trainingOutputs[i]);
            }
            totalError /= trainingInputs.Length;
            AdjustLearningRate(epochs);
            epochs++;
            if (epochs % 100 == 0)
            {
                Console.WriteLine($"Epoch: {epochs}, Error: {totalError}, Learning Rate: {currentLearningRate}");
            }
        }
        Console.WriteLine($"Training finished in {epochs} epochs with error: {totalError}");
    }

    public double[] Predict(double[] inputs)
    {
        return ForwardPropagation(inputs);
    }
}

public class DataGenerator
{
    private Random rand = new Random();

    public double[][] GenerateData(int dataSize, double minX, double maxX)
    {
        List<double[]> data = new List<double[]>();
        for (int i = 0; i < dataSize; i++)
        {
            double x = rand.NextDouble() * (maxX - minX) + minX;
            double fx = Math.Sin(2 * x);
            data.Add(new double[] { x, fx });
        }
        return data.ToArray();
    }
}

public class Program
{
    static void NormalizeData(double[][] data)
    {
        double[] mins = new double[data[0].Length];
        double[] maxs = new double[data[0].Length];
        for (int i = 0; i < data[0].Length; i++)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            for (int j = 0; j < data.Length; j++)
            {
                if (data[j][i] < min)
                    min = data[j][i];
                if (data[j][i] > max)
                    max = data[j][i];
            }
            mins[i] = min;
            maxs[i] = max;
        }
        for (int i = 0; i < data.Length; i++)
        {
            for (int j = 0; j < data[0].Length; j++)
            {
                data[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j]);
            }
        }
    }

    static void Main(string[] args)
    {
        Console.WriteLine("Enter the learning rate:");
        double learningRate = double.Parse(Console.ReadLine());
        Console.WriteLine("Enter the maximum epochs:");
        int maxEpochs = int.Parse(Console.ReadLine());
        Console.WriteLine("Enter the target error:");
        double targetError = double.Parse(Console.ReadLine());
        Console.WriteLine("Enter the size of the dataset:");
        int dataSize = int.Parse(Console.ReadLine());
        Console.WriteLine("Enter the minimum value for data range:");
        double minX = double.Parse(Console.ReadLine());
        Console.WriteLine("Enter the maximum value for data range:");
        double maxX = double.Parse(Console.ReadLine());
        Console.WriteLine("");
        Console.WriteLine("That is all we need! Press any key to continue...");
        Console.ReadLine();
        DataGenerator dataGenerator = new DataGenerator();
        double[][] trainingInputs = dataGenerator.GenerateData(dataSize, minX, maxX);
        double[][] trainingOutputs = dataGenerator.GenerateData(dataSize, minX, maxX);
        NormalizeData(trainingInputs);
        NormalizeData(trainingOutputs);
        double[][] testingInputs = dataGenerator.GenerateData(dataSize, minX, maxX);
        NormalizeData(testingInputs);
        NeuralNetwork neuralNetwork = new NeuralNetwork(1, 25, 1, learningRate);
        neuralNetwork.maxEpochs = maxEpochs;
        neuralNetwork.targetError = targetError;
        neuralNetwork.Train(trainingInputs, trainingOutputs);
        Console.WriteLine("");
        Console.WriteLine("Done training! Press any key to begin testing.");
        Console.ReadKey();
        List<double[]> predictions = new List<double[]>();
        for (int i = 0; i < testingInputs.Length; i++)
        {
            double[] prediction = neuralNetwork.Predict(testingInputs[i]);
            predictions.Add(new double[] { testingInputs[i][0], Math.Sin(2 * testingInputs[i][0]), prediction[0] });
        }
        if (File.Exists("result.csv"))
        {
            File.Delete("result.csv");
        }
        using (StreamWriter writer = new StreamWriter("result.csv"))
        {
            writer.WriteLine("x,f(x),Network Guess");
            foreach (var prediction in predictions)
            {
                writer.WriteLine($"{prediction[0]},{prediction[1]},{prediction[2]}");
            }
        }
        Console.WriteLine("Predictions written to result.csv");
        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();
    }
}
