//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100 /* 最大文字数は100字まで */
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000 /* 365行目と403行目で使用 */
#define MAX_CODE_LENGTH 40 /* 最大コード長は40字まで */

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

/* 構造体型struct vocab_wordを宣言 */
struct vocab_word {
  long long cn; /* long long型(64 bit符号付整数型) cn */
  int *point; /* int型ポインタpointを宣言 */
  char *word, *code, codelen; /* char型(1 byte文字型)codelenとポインタword，codeを宣言 */
}; /* この段階で変数無し */

char train_file[MAX_STRING], output_file[MAX_STRING]; /* 最大文字数MAX_STRINGを引数に持つchar型(1 byte文字型)train_fileとoutput_fileを宣言 */
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING]; /* char型save_vocab_fileとread_vocab_fileを宣言 */
struct vocab_word *vocab; /* 構造体型struct vocab_wordでポインタvocab(これは変数だからvocab.cn, vocab.point, vocab.word, vocab.code, vocab.codelenを持つ)を宣言 */
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1; /* スレッド数num_threads = 12，binary = 0は565行目で使用 */
int *vocab_hash; /* int型ポインタvocab_hash(SearchVocab()で使用) */
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100; /* long long型vocab_max_size, vocab_sizeとlayer1_sizeを宣言 */
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0; /* 560行目でclassesを使用) */
real alpha = 0.025, starting_alpha, sample = 1e-3; /* 655, 656行目よりalphaは学習比(learning rate) */
real *syn0, *syn1, *syn1neg, *expTable; /* real型ポインタ */
clock_t start;

int hs = 0, negative = 5; /* hsは344行目，negativeは350行目で初出 */
const int table_size = 1e8;
int *table; /* int型ポインタtable */
/* void型(値を返さない関数)InitUnigramTable()←かなり後に出てくる */
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int)); /* table_size * sizeof(int)分のメモリを動的に割り当て(このメモリはどこで解放？) */
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries /* ファイルから1単語読込み，スペース・タブ・行末を単語の切れ目と見なす */
void ReadWord(char *word, FILE *fin) { /* char型ポインタwordとファイルポインタfinが引数のvoid型関数ReadWord() */
  int a = 0, ch; /* int型aとch */
  while (!feof(fin)) { /* ファイルポインタfinがファイルの終端に達した時にループ終了 */
    ch = fgetc(fin); /* ファイルポインタfinから1文字読込んでint型で返す */
    if (ch == 13) continue; /* ch == 13の時処理をスキップ */
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {/* 空白・タブ・改行がある場合 */
      if (a > 0) { /* 最初は条件を満たさない */
        if (ch == '\n') ungetc(ch, fin); /* ファイルポインタfinに1文字返却しchを返す */
        break;
      }
      if (ch == '\n') { /* 改行が有る場合 */
        strcpy(word, (char *)"</s>"); /* 配列wordに文字列"</s>"をコピー */
        return;
      } else continue; /* 改行が無い場合処理をスキップ */
    }
    word[a] = ch; /* 配列wordのa番目にint型chを代入 */
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words /* 長すぎる単語を削除 */
  }
  word[a] = 0;
}

// Returns hash value of a word /* 単語のhash値を返す */
int GetWordHash(char *word) { /* char型ポインタwordを引数に持つint型関数GetWordHash */
  unsigned long long a, hash = 0; /* 符号無long long型a, hash */
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a]; /* aが文字列wordの文字長未満の時hashにhash * 257 + word[a]を代入 */
  hash = hash % vocab_hash_size; /* hashにhashをvocab_hash_sizeの剰余を代入 */
  return hash; /* hashを返す */
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1 /* 語彙中の単語の位置を返す(単語が語彙中に無い場合は-1を返す) */
int SearchVocab(char *word) { /* char型ポインタwordを引数に持つint型関数SearchVocab() */
  unsigned int hash = GetWordHash(word); /* 符号無int型hashにwordのhash値を代入 */
  while (1) { /* 無限ループ */
    if (vocab_hash[hash] == -1) return -1; /* vocab_hash[hash]が-1の時-1を返す */
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash]; /* 文字列wordとvocab[vocab_hash[hash]].wordが等しい時vocab_hash[hash]を返す */
    hash = (hash + 1) % vocab_hash_size; /* hashに(hash + 1) % vocab_hash_sizeを返す */
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary /* 単語を読取り，語彙中での単語の番号を返す */
int ReadWordIndex(FILE *fin) { /* ファイルポインタfinを引数に持つint型関数ReadWordIndex() */
  char word[MAX_STRING]; /* MAX_STRINGを引数に持つchar型wordを宣言 */
  ReadWord(word, fin); /* 先程定義したReadWordをchar型ポインタwordとファイルポインタfinを引数に計算 */
  if (feof(fin)) return -1; /* ファイルポインタが終端に達した時-1を返す */
  return SearchVocab(word); /* 単語wordの語彙中での位置を返す */
}

// Adds a word to the vocabulary /* 単語を語彙に加える */
int AddWordToVocab(char *word) { /* char型ポインタwordを引数に持つint型関数AddWordToVocab() */
  unsigned int hash, length = strlen(word) + 1; /* 符号無int型bash, length(wordの文字長+1を代入) */
  if (length > MAX_STRING) length = MAX_STRING; /* lengthが最大文字数より大きい場合はlenghに最大文字数を代入 */
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char)); /* length個のcharサイズのメモリを確保し，char型にしてvocab[vocab_size].wordに代入(158, 183行目で解放) */
  strcpy(vocab[vocab_size].word, word); /* vocab[vocab_size].wordにwordをコピー */
  vocab[vocab_size].cn = 0; /* vocab[vocab_size].cnに0を代入 */
  vocab_size++; /* vocab_sizeに1を足す */
  // Reallocate memory if needed /* 必要時にメモリの割当を変更 */
  if (vocab_size + 2 >= vocab_max_size) { /* vocab_size + 2がvocab_max_size以上の時 */
    vocab_max_size += 1000; /* vocab_max_sizeに1000を足す */
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word)); /*  */
  }
  hash = GetWordHash(word); /* hashにwordのhashを代入 */
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts /* 単語数で並替えをする際に使用 */
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts /* 語彙を単語数を用いて頻度順に並替え */
void SortVocab() { /* void関数SortVocab() */
  int a, size; 
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position /* 語彙を並替えて文字列</s>を先頭に保つ */
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);/* vocab[1]のアドレス，vocab_size -1， */
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab /* min_count以下の頻度の単語をvocabから除外する */
    if ((vocab[a].cn < min_count) && (a != 0)) { /* vocab[a].cnがmin_count未満かつaが0でない時 */
      vocab_size--; /* vocab_sizeから1引く */
      free(vocab[a].word); /* 125行目で確保したvocab[a].wordのメモリを解放 */
    } else {
      // Hash will be re-computed, as after the sorting it is not actual /* ハッシュを */
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word); /* 125行目で確保したvocab[a].wordのメモリを解放 */
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts /* 語数を用いて2値Huffman木を作成 */
// Frequent words will have short uniqe binary codes /* 高頻度の単語に短い一意的な2進数を割当る */
void CreateBinaryTree() { /* void関数CreateBinaryTree() */
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH]; /* char型code[最大コード長MAX_CODE_LENGTH] */
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long)); /* long long型ポインタcountに(vocab_size * 2 * 1) * (long long)分のメモリを確保 */
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long)); /* long long型ポインタbinaryに(vocab_size * 2 * 1) * (long long)分のメモリを確保 */
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long)); /* long long型ポインタparent_nodeに(vocab_size * 2 * 1) * (long long)分のメモリを確保 */
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn; /* ポインタcountの0 <= a < vocab_size番目ににvocab[a].cnを代入 */
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15; /* ポインタcountのvocab_size <= a < vocab_size * 2番目ににvocab[a].cnを代入 */
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time /* ノードをa回追加してHuffman木を構成するアルゴリズム */
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2' /* まず2つの最小ノード'min1, min2'を探す */
    if (pos1 >= 0) { /* pos1(最初はvocab_size-1)が非負の時 */
      if (count[pos1] < count[pos2]) { /* count[pos1] < count[pos2](最初はpos1 = vocab_size - 1, pos2 = vocab_size)の時 */
        min1i = pos1; /* long long型min1iにpos1を代入 */
        pos1--;
      } else { /* count[pos1] >= count[pos2]の時 */
        min1i = pos2;
        pos2++;
      }
    } else { /* pos1が負の時 */
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) { /* count[pos1] < count[pos2](最初はpoos1 = vocab_size - 1, pos2 = vocab_size)の時 */
        min2i = pos1; /* long long型min2iにpos1を代入 */
        pos1--;
      } else { /* count[pos1] >= count[pos2]の時 */
        min2i = pos2;
        pos2++;
      }
    } else { /* pos1が負の時 */
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i]; /* count[vocab_size + a]にcount[min1i] + count[min2i]を代入 */
    parent_node[min1i] = vocab_size + a; /* min1iの親ノードにvocab_size + a を代入 */
    parent_node[min2i] = vocab_size + a; /* min2iの親ノードにvocab_size + a を代入 */
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word /* 語彙の各単語に二進法コードを割当 */
  for (a = 0; a < vocab_size; a++) { /* 0 <= a < vocab_sizeの時 */
    b = a;
    i = 0;
    while (1) { /* 無限ループ */
      code[i] = binary[b]; /* code[i]にbinary[b](最初はb=a)を代入 */
      point[i] = b; /* pont[i]にbを代入*/
      i++;
      b = parent_node[b]; /* bにparent_nodeを代入 */
      if (b == vocab_size * 2 - 2) break; /* b == vocab_size * 2 - 2の時無限ループから脱出 */
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count); /* count分のメモリを解放 */
  free(binary); /* binary分のメモリを解放 */
  free(parent_node); /* parent_node分のメモリを解放 */
}

void LearnVocabFromTrainFile() { /* void型関数LearnVocabFromTrainFile()，297行目まで */
  char word[MAX_STRING];
  FILE *fin; /* FILE型ポインタfin */
  long long a, i; /* long long型変数a, i */
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; /* 0 <= a < vocab_hash_sizeの時vocab_hash[a]に-1を代入 */
  fin = fopen(train_file, "rb"); /* train_fileをバイナリモードで読出専用で開く */
  if (fin == NULL) { /* ポインタfinがNULLの時 */
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>"); /* 単語を語彙に加える関数AddWordToVocab */
  while (1) { /* 無限ループ */
    ReadWord(word, fin); /* 72行目で定義したReadWord関数でtrain_fileの単語を読み込む */
    if (feof(fin)) break; /* ファイルポインタが終端に達した時無限ループから脱出 */
    train_words++; /* train_wordsに1を足す */
    if ((debug_mode > 1) && (train_words % 100000 == 0)) { /* debug_mode > 1かつtrain_wordsが100000で割切れる時 */
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word); /* 103行目で定義した語彙中の単語の位置を返す関数SearchVocab */
    if (i == -1) { /* i == -1の時 */
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab(); /* 146行目で定義した語彙を単語数を用いて頻度順に並替える関数SortVocab() */
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin); 
} /* 263行目からのLearnVocabFromTrainFile終わり */

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb"); /* save_vocab_fileをバイナリモードで書込専用で開く */
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() { /* void型関数ReadVocab，337行目まで */
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb"); /* read_vocab_fileをバイナリモードで読出専用で開きファイルポインタ*finに代入 */
  if (fin == NULL) { /* finがNULLの時 */
    printf("Vocabulary file not found\n");
    exit(1); /* 異常終了 */
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; /* 0 <= a < vocab_hash_sizeの時 */
  vocab_size = 0;
  while (1) { /* 無限ループ */
    ReadWord(word, fin); /* 72行目で定義したReadWord関数 */
    if (feof(fin)) break; /* ファイルポインタが終端に達した時無限ループから脱出 */
    a = AddWordToVocab(word); /* 単語を語彙に加える関数AddWordToVocab */
    fscanf(fin, "%lld%c", &vocab[a].cn, &c); /* ファイルポインタfinから読出した値をvocab[a].cnに取込み， */
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb"); /* 読出モードでtrain_fileを開く */
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END); /* ファイルfinのファイル位置演算子をSEEK_ENDを基準に0バイト移動 */ 
  file_size = ftell(fin);
  fclose(fin);
} /* 306行目からのReadVocab終わり */

void InitNet() { /* void関数IniNet() */
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real)); /*  */
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) { /* hs=0よりこのif文はFALSE */
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real)); /* (long long)vocab_size * layer1_size * sizeof(real) bytesのメモリを割当て，割当たメモリのアドレスを(void **)&syn1に割当る．割当たメモリのアドレスは128の倍数 */
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);} /* 上でメモリの割当に失敗した時に表示 */
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) { /* negative=5よりこのif文はTRUE */
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real)); /* (long long)vocab_size * layer1_size * sizeof(real) bytesのメモリを割当て，割当たメモリのアドレスを(void **)&syn1negに割当る．割当たメモリのアドレスは128の倍数 */
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);} /* 上でメモリの割当に失敗した時に表示 */
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) /* 0 <= a < vocab_size & 0 <= b < layer1_sizeの時 */
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree(); /* 198行目で定義したCreateBinaryTree()でHuffman木を生成 */
}

void *TrainModelThread(void *id) { /* 543行目まである */
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1]; /* 配列senは1001個の要素を持つ */
  long long l1, l2, c, target, label, local_iter = iter; /* 44行目よりiter == 5 */
  unsigned long long next_random = (long long)id;
  real f, g; /* real型変数f, g (skip-gram等で頻出) */
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real)); /* *neu1にメモリを動的に割当， 540行目で解放*/
  real *neu1e = (real *)calloc(layer1_size, sizeof(real)); /* *neu1eにメモリを動的に割当，541行目で解放 */
  FILE *fi = fopen(train_file, "rb"); /* 読出モードでtrain_fileを開き，ファイルポインタfiに代入，374行目から538行目までの無限ループの後539行目で閉じる */
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);  /* ファイルfiのファイル位置演算子をSEEK_SETを基準にfile_size / (long long)num_threads * (long long)idバイト移動 */
  while (1) { /* 無限ループ(538行目まで) */
    if (word_count - last_word_count > 10000) { /* 387行目まで */
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) { /* 384行目まで */
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) { /* if文(406行目まで),364行目より当初はsentence_length == 0だが389行目からの無限ループで最終的にsentence_lengthが1000になる */
      while (1) { /* 無限ループ(404行目まで) */
        word = ReadWordIndex(fi); /* 114行目で定義の単語を読取り，語彙中での単語の番号を返すReadWordIndex */
        if (feof(fi)) break; /* ファイルポインタが終端に達した時389行目からの無限ループから脱出 */
        if (word == -1) continue; /* word == -1の時処理をスキップ */
        word_count++;
        if (word == 0) break; /* 389行目からの無限ループから脱出 */
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) { /* sample > 0 の時(400行目まで) */
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue; /* ran < (next_random & 0xFFFF) / (real)65536 の時処理をスキップして390行目に戻る */
        } /* 396行目からのif文ここまで */
        sen[sentence_length] = word; /* 配列senの各要素にwordを代入 */
        sentence_length++; /* ここでsentence_lengthが増える */
	if (sentence_length >= MAX_SENTENCE_LENGTH) break; /* sentence_lengthがMAX_SENTENCE_LENGTH == 1000に達した時389行目からの無限ループから脱出 (i.e. 配列senはMAX_SENTENCE_LENGTH + 1個の要素が有るので，senの各要素にwordを代入し終わったら無限ループを抜ける) */
      } /*  389行目からの無限ループここまで */
      sentence_position = 0; /* 364行目でlong long sentence_position = 0と置いているが，ここでも改めて代入(533行目のsentence_position++はここに響く？) */
    } /* 388行目から */
    if (feof(fi) || (word_count > train_words / num_threads)) { /* ファイルポインタが終端に達した若くはword_countがtrain_words / num_threadsより多い場合(416行目まで) */
      word_count_actual += word_count - last_word_count;
      local_iter--; /* local_iterを1つ減らす(366行目より当初はlocal_iter == 5) */
      if (local_iter == 0) break; /* ここで374行目からの無限ループから脱出！ */
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);  /* ファイルfiのファイル位置演算子をSEEK_SETを基準にfile_size / (long long)num_threads * (long long)idバイト移動 */
      continue; /* 以下の処理をスキップして375行目に一気に戻る */
    } /* 407行目から */
    word = sen[sentence_position]; /* 最初はword = sen[0] */
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11; /* 線形合同法で乱数next_randomを生成 */
    b = next_random % window; /* 41行目よりwindow = 5, このbは483行目以降のelseで使う */
    if (cbow) {  //train the cbow architecture /* 連続単語集合モデルCBOWの学習(41行目からcbow == 1よりデフォルトだとこれが動く)ここから482行目まで読み飛ばす */
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else {  //train skip-gram /* skip-gramの学習(41行目からcbow == 1よりデフォルトだと動かない！)，532行目まで */
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) { /* 531行目まで，(422行目より next_random % window =) b <= a < window *2 + 1 -bの間で，a != windowの時 */
        c = sentence_position - window + a; /* 41行目よりwindow == 5 */
        if (c < 0) continue; /* c < 0の時処理をスキップして485行目に戻る */
        if (c >= sentence_length) continue; /* c >= sentence_lengthの時処理をスキップして485行目に戻る(384行目より当初はsentence_lengh == 0) */
        last_word = sen[c];
        if (last_word == -1) continue; /* last_word == -1の時処理をスキップ */
        l1 = last_word * layer1_size; /* 366行目で定義したl1にlast_word * layer1_sizeを代入 */
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX /* 階層的ソフトマックス */
        if (hs) for (d = 0; d < vocab[word].codelen; d++) { /* 507行目まで(デフォルトだとhs == 0だから動かない！) */
          f = 0; /* real型変数f(368行目で宣言済)に0を代入 */
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output /* hiddenからoutputに反映 */
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue; /* f <= -MAX_EXP(== -6, 23行目)の時処理をスキップ */
          else if (f >= MAX_EXP) continue; /* f >= MAX_EXP(== 6, 23行目)の時処理をスキップ */
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]; /* -MAX_EXP < f < MAX_EXPの時 fにexpTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]を代入 */
          // 'g' is the gradient multiplied by the learning rate /* 'g'は勾配と学習比の積 */
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden /* エラーをoutputからhiddenに反映 */
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output /* 重みを学習してhidenから */
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        } /* 493行目から */
        // NEGATIVE SAMPLING /* ネガティブサンプリング */
        if (negative > 0) for (d = 0; d < negative + 1; d++) { /* 528行目まで(デフォルトだとnegative == 5だからこっちが動く) */
          if (d == 0) { /* ループの最初だけ */
            target = word;
            label = 1;
          } else { /* ループの2回目以降 */
            next_random = next_random * (unsigned long long)25214903917 + 11; /* 線形合同法 */
            target = table[(next_random >> 16) % table_size]; /* targetにtable[(next_random >> 16) % table_size]を代入 */
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue; /* target == wordの時処理をスキップして510行目に戻る */
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        } /* 509行目からのfor文はここまで */
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    } /* 483行目からのskip-gram終わり */
    sentence_position++; /* sentence_positionはここで1増える */
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue; /* sentence_position >= sentence_lengthの時sentence_length = 0とした上で処理をスキップし，375行目に戻る */
    }
  } /* 374行目から無限ループ終り */
  fclose(fi); /* 372行目で開いたtrain_fileを閉じる */
  free(neu1); /* 370行目で確保したneu1のメモリを解放 */
  free(neu1e); /* 371行目で確保したneu1eのメモリを解放 */
  pthread_exit(NULL); /* 呼び出したスレッドを終了(557行目と558行目に関係) */
} /* 363行目からのTrainModelThread終わり */

void TrainModel() { /* 614行目まで */
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t)); /* pthread(マルチスレッドのライブラリ) */
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile(); /* read_vocab_file[0] != 0の時は306行目のReadVocab，そうでない時は263行目のLearnVocabFromTrainFile */
  if (save_vocab_file[0] != 0) SaveVocab(); /* save_vocab_file[0] != 0の時は299行目のSaveVocab() */
  if (output_file[0] == 0) return; /* output_file[0] == 0の時はTrainModelはここで終わり */
  InitNet(); /* 339行目のInitNet */
  if (negative > 0) InitUnigramTable(); /* デフォルトではnegative == 5よりTRUE，53行目のInitUnigramTable */
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a); /* 0 <= 1 < num_threadsで新規にスレッド(ID: pt[a])を作成し，TrainModelThreadを実行する */
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL); /* スレッドID: pt[a]が終了(TrainModelThreadが543行目でスレッドを終わらせる)するまで実行を一時停止 */
  fo = fopen(output_file, "wb"); /* output_fileをバイナリモードで書込専用で開く，613行目で閉じる */
  if (classes == 0) { /* デフォルトでは44行目のclasses = 0よりTRUE，569行目まで */
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size); /* foにvocab_sizeとlayer1_sizeをlong long型でそれぞれ出力 */
    for (a = 0; a < vocab_size; a++) { /* 0 <= a < vocab_sizeの時，568行目まで */
      fprintf(fo, "%s ", vocab[a].word); /* foにvocab[a].wordを文字列として出力 */
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo); /* デフォルトでは41行目のbinary = 0より動く，0 <= b < layer1_sizeの時foにポインタsyn0[a * layer1_size + b]からsizeof(real) bytes単位で1個のデータを書込み */
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]); /* デフォルトでは動かない */
      fprintf(fo, "\n"); /* foに改行を出力 */
    } /* 563行目からのfor文ここまで */
  } else { /* 612行目まで，デフォルトではclasses == 0より動かない！ */
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int)); /* *centcnに動的にメモリ割当，609行目で解放 */
    int *cl = (int *)calloc(vocab_size, sizeof(int)); /* *clに動的にメモリ割当，611行目で解放 */
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real)); /* *centに動的にメモリ割当，610行目で解放 */
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn); /* 572行目で確保したcentcnのメモリを解放 */
    free(cent); /* 575行目で確保したcentのメモリを解放 */
    free(cl); /* 573行目で確保したclのメモリを解放 */
  } /* 569行目elseから */
  fclose(fo); /* 559行目で開いたfoを閉じる */
} /* 545行目void TrainModel()から */

int ArgPos(char *str, int argc, char **argv) { /* 676行目以下に頻出 */
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) { /* 文字列strとargv[a]が等しい時 */
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1); /* 1 <= a < argcかつ文字列str==argv[a]の時にfor文が動き，a == argc - 1の時にメッセージを表示，処理失敗で終了(これが成立するのは実引数がstrだけの場合と，実引数の最後がstrの時，i.e. -<argument>とだけ書いて直後に数字を書かないケースを排除) */
    }
    return a; /* 1 <= a < argc -1かつ文字列str==argv[a]の時だけここに来る */
  } /* 618行目から */
  return -1; /* 実引数が無い時や文字列strに一致するargv[a]が無い時にここに来る  */
} /* 616行目から */

int main(int argc, char **argv) { /* main関数，703行目まで(argc:コマンドライン引数の総個数-1(何も引数をつけない時はargc==1)，argv: コマンドライン引数の文字列を指すポインタの配列 */
  int i;
  if (argc == 1) { /* コマンドライン引数に何も付けない時，672行目まで */
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  } /* 630行目からのif文終わり */
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]); /* 実引数"-size"が直後の数字と共に書かれた場合に限り直後の数字atoi(argv[i + 1]をint型に変換してlayer1_sizeに代入) */
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]); /* 実引数"-train"が直後の数字と共に書かれた場合に限りtrain_fileに直後の数字argv[i + 1]をコピー */
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]); /* 実引数"-save-vocab"が直後の数字と共に書かれた場合に限りsave_vocab_fileに直後の数字argv[i + 1]をコピー */
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]); /* 実引数"-read-vocab"が直後の数字と共に書かれた場合に限りread_vocab_fileに直後の数字argv[i + 1]をコピー */
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]); /* 実引数"-debug"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をint型に変換してdebug_modeに代入 */
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]); /* 実引数"-binary"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をint型に変換してbinaryに代入 */
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]); /* 実引数"-cbow"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をint型に変換してcbowに代入 */
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]); /* 実引数"-alpha"が直後の数字と共に書かれた場合に限りargv[i + 1]をdouble型に変換してalphaに代入 */
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]); /* 実引数"-output"が直後の数字と共に書かれた場合に限りoutput_fileに直後の数字argv[i + 1]をコピー */
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]); /* 実引数"-window"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をint型に変換してwindowに代入 */
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]); /* 実引数"-sample"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をdouble型に変換してsampleに代入 */
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]); /* 実引数"-hs"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をint型に変換してhsに代入 */
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]); /* 実引数"-negative"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をint型に変換してnegativeに代入 */
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]); /* 実引数"-threads"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をint型に変換してnum_threadsに代入 */
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]); /* 実引数"-iter"が直後の数字と共に書かれた場合に限りargv[i + 1]をint型に変換してiterに代入 */
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]); /* 実引数"-min-count"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をint型に変換してmin_countに代入 */
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]); /* 実引数"-classes"が直後の数字と共に書かれた場合に限り直後の数字argv[i + 1]をint型に変換してclassesに代入 */
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word)); /* vocabに動的にメモリ割当(どこで解放？) */
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int)); /* vocab_hashに動的にメモリ(vocab_hash_size = 30M)割当(どこで解放？) */
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real)); /* expTableに動的にメモリ割当(どこで解放？) */
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel(); /* 545行目で定義したTrainModelを開く */
  return 0;
}
