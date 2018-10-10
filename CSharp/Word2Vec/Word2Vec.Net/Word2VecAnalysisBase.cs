using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Word2Vec.Net.Utils;

namespace Word2Vec.Net
{
    public class Word2VecAnalysisBase
    {
        public const long max_size = 2000;         // max length of strings
        public const long N = 40;                  // number of closest words that will be shown
        public const long max_w = 50;              // max length of vocabulary entries
        private long size;
        private string file_name;
        private long words;
        private char[] vocab;
        private float[] m;
        protected char[] Vocab
        {
            get
            {
                return vocab;
            }

            set
            {
                vocab = value;
            }
        }

        protected long Words
        {
            get
            {
                return words;
            }

            set
            {
                words = value;
            }
        }

        protected long Size
        {
            get
            {
                return size;
            }

            set
            {
                size = value;
            }
        }

        protected float[] M
        {
            get
            {
                return m;
            }

            set
            {
                m = value;
            }
        }

        /// <summary>
        /// Basic class for analysis algorithms( distnace, analogy, commpute-accuracy)
        /// </summary>
        /// <param name="fileName"></param>
        public Word2VecAnalysisBase(string fileName)
        {
            file_name = fileName;           //bestw = new string[N];



            InitVocub();
        }

        private void InitVocub()
        {
            var lines = File.ReadAllLines(file_name).ToList();
            Words = lines.Count();
            Size = 50;
            lines = lines.Skip(0).ToList();
            M = new float[Words * Size];
            var allWords = lines.Select(x => x.Split(' ')[0]).ToList();

            Vocab = new char[Words * max_w];
            for (int b = 1; b < Words; b++)
            {
                int a = 0;
                int i = 0;

                string word = allWords[b];

                foreach (char ch in word)
                {
                    Vocab[b * max_w + a] = ch;
                    if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
                }
                Vocab[b * max_w + a] = '\0';

                var allValues = lines[b].Trim().Split(' ').Skip(1).Select(x => float.Parse(x)).ToList();
                for (a = 0; a < Size; a++)
                {
                    byte[] bytes = new byte[8];
                    M[a + b * Size] = allValues[a];
                }
                float len = 0;
                for (a = 0; a < Size; a++) len += M[a + b * Size] * M[a + b * Size];
                len = (float)Math.Sqrt(len);
                for (a = 0; a < Size; a++) M[a + b * Size] = M[a + b * Size] / len;
            }
        }
    }
}
