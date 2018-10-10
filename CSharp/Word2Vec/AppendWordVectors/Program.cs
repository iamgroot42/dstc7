using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
namespace ConsoleApp3
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var word2vecLines = File.ReadAllLines("Custom.50d.txt").ToList();
            Dictionary<string, Tuple<float[], float[]>> dict = new Dictionary<string, Tuple<float[], float[]>>();
            int gloveSize = 300;
            int word2VecSize = 50;

            StreamReader sr = new StreamReader("glove.840B.300d.txt");
            string line1;
            int cnter = 0;
            while (true)
            {
                cnter++;
                line1 = sr.ReadLine();
                if (string.IsNullOrEmpty(line1)) break;

                if (cnter % 100000 == 0)
                    Console.WriteLine(cnter);
                string word = line1.Split(' ')[0];
                float[] value = new float[gloveSize];
                value = line1.Split(' ').Skip(1).Select(x => float.Parse(x)).ToArray();
                dict[word] = new Tuple<float[], float[]>(value, new float[word2VecSize]);
            }

            foreach (var line in word2vecLines.Skip(1))
            {
                string word = line.Split(' ')[0];
                float[] value = new float[word2VecSize];
                value = line.Split(' ').Skip(1).Where(x => !string.IsNullOrWhiteSpace(x)).Select(x => float.Parse(x.Trim())).ToArray();

                if (dict.ContainsKey(word))
                {
                    dict[word] = new Tuple<float[], float[]>(dict[word].Item1, value);
                }
                else
                {
                    dict[word] = new Tuple<float[], float[]>(new float[gloveSize], value);
                }
            }

            StreamWriter sw = new StreamWriter("CasedGlove" + gloveSize + "_custom" + word2VecSize + ".txt");
            int i = 0;
            int cnt = 0;
            foreach (var kv in dict)
            {
                cnt++;
                var allValues = kv.Value.Item1.ToList();
                allValues.AddRange(kv.Value.Item2);

                if (cnt % 100000 == 0)
                {
                    Console.WriteLine(cnt);
                }

                if (allValues.Count != gloveSize + word2VecSize)
                    Console.WriteLine("error");
                sw.WriteLine(kv.Key + ' ' + string.Join(' ', allValues));
            }

            sw.Close();
        }
    }
}
