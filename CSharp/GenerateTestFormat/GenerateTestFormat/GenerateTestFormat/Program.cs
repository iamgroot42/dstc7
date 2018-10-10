using System;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GenerateForTest
{
    static class Program
    {
        static void Main(string[] args)
        {
            int total = 0;

            List<TestInstance> final = new List<TestInstance>();
            using (StreamReader sr = new StreamReader(args[0]))
            {
                using (JsonReader reader = new JsonTextReader(sr))
                {
                    JsonSerializer serializer = new JsonSerializer();
                    var result = serializer.Deserialize<List<Instance>>(reader);

                    var lines = File.ReadAllLines(args[1]);
                    for (int i = 0; i < lines.Length; i++)
                    {
                        var tempList = result[i].optionsfornext.Select((x, temp) => new KeyValuePair<Candidate, int>(x, temp));

                        var scores = lines[i].Split(',').Select(x => double.Parse(x));

                        var sorted = scores.Select((x, temp) => new KeyValuePair<double, int>(x, temp)).OrderByDescending(x => x.Key).ToList();

                        total++;

                        TestInstance test = new TestInstance(result[i].exampleid);

                        for (int j = 0; j < 100; j++)
                        {
                            test.candidateranking.Add(new TestCandidate(scores.ToList()[j], tempList.ToList()[j].Key.candidateid));
                        }

                        test.candidateranking = test.candidateranking.OrderByDescending(x => x.confidence).ToList();

                        test.candidateranking.ForEach(x => x.confidence = Math.Round(x.confidence, 3));
                        final.Add(test);
                    }


                    StreamWriter sw = new StreamWriter(args[2]);
                    sw.Write(JsonConvert.SerializeObject(final, Formatting.Indented));

                    sw.Close();
                }
            }

            Console.WriteLine(total);
            Console.ReadLine();
        }        
    }

    class TestInstance
    {
        [JsonProperty(PropertyName = "example-id")]
        public string exampleid;

        [JsonProperty(PropertyName = "candidate-ranking")]
        public List<TestCandidate> candidateranking;

        public TestInstance(string exampleid)
        {
            this.exampleid = exampleid;
            this.candidateranking = new List<TestCandidate>();
        }
    }

    class Instance
    {
        [JsonProperty(PropertyName = "data-split")]
        public string datasplit;

        [JsonProperty(PropertyName = "example-id")]
        public string exampleid;

        [JsonProperty(PropertyName = "messages-so-far")]
        public List<Message> messages;

        [JsonProperty(PropertyName = "options-for-correct-answers")]
        public List<Candidate> optionsforcorrect;

        [JsonProperty(PropertyName = "options-for-next")]
        public List<Candidate> optionsfornext;

        public string dssm;
    }

    class Message
    {
        public string speaker;
        public string utterance;
    }

    class Candidate
    {
        [JsonProperty(PropertyName = "candidate-id")]
        public string candidateid;
        public string utterance;
    }

    class TestCandidate
    {
        [JsonProperty(PropertyName = "candidate-id")]
        public string candidateid;

        [JsonProperty(PropertyName = "confidence")]
        public double confidence;

        public TestCandidate(double confidence, string candidateId)
        {
            this.confidence = confidence;
            this.candidateid = candidateId;

        }

        public TestCandidate()
        {

        }
    }
}
