import re
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def process_html_file(file_path, color_code='#4c82a3ff'):
    """
    Process an HTML file to extract words from <span> tags with a specific color style.

    Args:
    file_path (str): Path to the HTML file.
    color_code (str): Hex color code to search for in the <span> tags (default is '#4c82a3ff').

    Returns:
    dict: Word count dictionary sorted by frequency.
    """
    # Step 1: Read the content of the HTML file into a string
    with open(file_path, 'r') as f:
        html = f.read()

    # Step 2: Convert HTML to lowercase for consistent matching
    html = html.lower()

    # Step 3: Regex to match: <span style='color: #4c82a3ff'>word</span>
    pattern = rf"<span style='color: {color_code}'>(.*?)</span>"

    # Step 4: Extract all matched words
    matches = re.findall(pattern, html)

    # Step 5: Count the occurrences of each word
    word_count = defaultdict(int)
    for word in matches:
        word_count[word] += 1

    # Step 6: Sort the dictionary by count in descending order
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    # Step 7: Clean spaces from keys while maintaining the sorted order
    sorted_word_count = [(k.replace(' ', ''), v) for k, v in sorted_word_count]

    # Convert to dict only at the end (for wordcloud compatibility)
    return dict(sorted_word_count)


CORE_DIR = "/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals"
# for each item in CORE_DIR/data/wordcloud_data
wordcloud_data_path = os.path.join(CORE_DIR, 'data', 'wordcloud_data')
# first sort alphabetically
sorted_items = sorted(os.listdir(wordcloud_data_path))
for item in sorted_items:
    item_path = os.path.join(wordcloud_data_path, item)
    
    if os.path.isfile(item_path) and item.endswith('.html'):
        # Process single HTML file
        wcd = process_html_file(item_path)
        # Print in sorted order
        print(f"## CATEGORY: {item.split('.')[0]} ##\n")
        for word, count in sorted(wcd.items(), key=lambda x: x[1], reverse=True):
            print(f"{word}: {count}")
        wordcloud = WordCloud(width=600, height=400, background_color='white', 
                            max_words=50, colormap='gnuplot2_r').generate_from_frequencies(wcd)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(os.path.splitext(item)[0] + ' vocabulary\n', fontsize=15)
        plt.show()

