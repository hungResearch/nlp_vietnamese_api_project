import argparse
import re
from typing import List
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize data.'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help='File to input',
    )
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['text', 'label'],
        help='File to input',
    )
    return parser.parse_args()


def read_training_file(path: str) -> List[str]:
    handler = open(path, 'r')
    for line in handler.readlines():
        yield line.strip()
    handler.close()


def main():
    args = parse_arguments()
    lines = read_training_file(args.train_file)
    documents = []
    labels = []
    for line in lines:
        parts = line.split('\t')
        documents.append(parts[0])
        labels.append(parts[1])

    if args.type == 'text':
        plt.title('Text Statistics')
        text = ' '.join(documents)
        wordcloud = WordCloud(
            background_color="white",
            max_words=len(text),
            max_font_size=80,
            relative_scaling=.9
        ).generate(text)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
    if args.type == 'label':
        plt.title('Label Statistics')
        label_set = list(set(labels))
        counts = []
        for label in label_set:
            cnt = 0
            for l in labels:
                if l == label:
                    cnt += 1
            counts.append(cnt)
        plt.pie(counts, labels=label_set, autopct='%1.1f%%')
        plt.show()



if __name__ == "__main__":
    main()
