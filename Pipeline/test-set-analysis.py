import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter, defaultdict
import glob
from tqdm import tqdm

# Paths to directories
TEST_SET_DIR = '/Users/pranavkhetarpal/Desktop/M3RG/Composition-Extraction/test-set-OG/'
PROMPTING_DATA_DIR = '/Users/pranavkhetarpal/Desktop/M3RG/Composition-Extraction/Pipeline/prompting_data/'
RESEARCH_PAPER_TEXT_DIR = os.path.join(PROMPTING_DATA_DIR, 'research-paper-text/')
RESEARCH_PAPER_TABLES_DIR = os.path.join(PROMPTING_DATA_DIR, 'research-paper-tables/')
MATSKRAFT_TABLES_DIR = os.path.join(PROMPTING_DATA_DIR, 'Matskraft-tables/')

# Ensure output directory exists
os.makedirs('analysis_results', exist_ok=True)

def load_test_set():
    """Load the gold test set data"""
    json_files = [f for f in os.listdir(TEST_SET_DIR) if f.endswith('.json')]
    test_data = {}
    
    for filename in tqdm(json_files, desc="Loading gold test set"):
        if filename == '.DS_Store':
            continue
        
        paper_id = filename.replace('.json', '')
        try:
            with open(os.path.join(TEST_SET_DIR, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_data[paper_id] = data
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    return test_data

def get_paper_content(paper_id):
    """Get the text content of a paper"""
    text_filename = f"{paper_id}.txt"
    text_path = os.path.join(RESEARCH_PAPER_TEXT_DIR, text_filename)
    
    try:
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Warning: Could not read text file for {paper_id}: {str(e)}")
        return None

def get_table_content(paper_id):
    """Get the table content of a paper"""
    table_filename = f"{paper_id}.txt"
    table_path = os.path.join(RESEARCH_PAPER_TABLES_DIR, table_filename)
    
    try:
        with open(table_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Warning: Could not read table file for {paper_id}: {str(e)}")
        return None

def analyze_compositions(test_data):
    """Analyze the extracted compositions"""
    all_compositions = []
    compositions_per_paper = {}
    element_counts = Counter()
    formula_lengths = []
    
    # Regular expression to identify elements in a formula
    element_pattern = re.compile(r'([A-Z][a-z]*)(\d*\.?\d*)|(\d+\.?\d*)')
    
    for paper_id, data in test_data.items():
        try:
            compositions = data['response']['Compositions']
            compositions_per_paper[paper_id] = len(compositions)
            
            for item in compositions:
                composition = item.get('composition', '')
                if composition:
                    all_compositions.append(composition)
                    formula_lengths.append(len(composition))
                    
                    # Extract elements
                    for match in element_pattern.finditer(composition):
                        element = match.group(1)
                        if element and len(element) <= 2:  # Most element symbols are 1-2 chars
                            element_counts[element] += 1
        except Exception as e:
            print(f"Error analyzing compositions for {paper_id}: {str(e)}")
    
    return {
        'total_compositions': len(all_compositions),
        'unique_compositions': len(set(all_compositions)),
        'compositions_per_paper': compositions_per_paper,
        'avg_compositions_per_paper': np.mean(list(compositions_per_paper.values())),
        'median_compositions_per_paper': np.median(list(compositions_per_paper.values())),
        'max_compositions_per_paper': max(compositions_per_paper.values()),
        'formula_length_stats': {
            'mean': np.mean(formula_lengths),
            'median': np.median(formula_lengths),
            'min': min(formula_lengths),
            'max': max(formula_lengths)
        },
        'top_elements': element_counts.most_common(10),
        'all_elements': element_counts
    }

def analyze_notation_styles(all_compositions):
    """Analyze notation styles in compositions"""
    # Define patterns for different notation styles
    notation_patterns = {
        'subscript_numbers': re.compile(r'[A-Z][a-z]?\d+'),
        'decimal_subscripts': re.compile(r'[A-Z][a-z]?\d+\.\d+'),
        'variables': re.compile(r'[A-Z][a-z]?[xyz]'),
        'ranges': re.compile(r'[A-Z][a-z]?\d+-\d+'),
        'parentheses': re.compile(r'\([^)]+\)'),
        'square_brackets': re.compile(r'\[[^\]]+\]')
    }
    
    notation_counts = {style: 0 for style in notation_patterns}
    
    for composition in all_compositions:
        for style, pattern in notation_patterns.items():
            if pattern.search(composition):
                notation_counts[style] += 1
    
    return notation_counts

def paper_type_distribution(test_data):
    """Analyze paper types based on patterns in their content"""
    paper_types = {
        'battery_related': 0,
        'magnetic_materials': 0,
        'semiconductors': 0,
        'catalysts': 0,
        'other': 0
    }
    
    # Keywords for classification
    keywords = {
        'battery_related': ['battery', 'electrolyte', 'cathode', 'anode', 'energy storage'],
        'magnetic_materials': ['magnetic', 'ferromagnetic', 'ferroelectric'],
        'semiconductors': ['semiconductor', 'photovoltaic', 'solar cell', 'bandgap'],
        'catalysts': ['catalyst', 'catalytic', 'photocatalyst']
    }
    
    for paper_id in test_data:
        content = get_paper_content(paper_id)
        if not content:
            continue
        
        content = content.lower()
        paper_classified = False
        
        for paper_type, kw_list in keywords.items():
            if any(kw in content for kw in kw_list):
                paper_types[paper_type] += 1
                paper_classified = True
                break
        
        if not paper_classified:
            paper_types['other'] += 1
    
    return paper_types

def analyze_test_set():
    """Main analysis function"""
    print("Starting test set analysis...")
    
    # Load data
    test_data = load_test_set()
    print(f"Loaded {len(test_data)} papers from test set")
    
    # Analyze compositions
    print("Analyzing compositions...")
    composition_stats = analyze_compositions(test_data)
    
    # Get all compositions
    all_compositions = []
    for paper_id, data in test_data.items():
        try:
            compositions = data['response']['Compositions']
            for item in compositions:
                comp = item.get('composition', '')
                if comp:
                    all_compositions.append(comp)
        except:
            continue
    
    # Analyze notation styles
    print("Analyzing notation styles...")
    notation_stats = analyze_notation_styles(all_compositions)
    
    # Paper type distribution
    print("Analyzing paper types...")
    paper_types = paper_type_distribution(test_data)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Compositions per paper histogram
    plt.figure(figsize=(10, 6))
    plt.hist(list(composition_stats['compositions_per_paper'].values()), bins=20)
    plt.xlabel('Number of Compositions')
    plt.ylabel('Number of Papers')
    plt.title('Compositions per Paper')
    plt.savefig('analysis_results/compositions_per_paper.png')
    plt.close()
    
    # Formula length histogram
    plt.figure(figsize=(10, 6))
    formula_lengths = [len(comp) for comp in all_compositions]
    plt.hist(formula_lengths, bins=20)
    plt.xlabel('Formula Length (characters)')
    plt.ylabel('Count')
    plt.title('Distribution of Formula Lengths')
    plt.savefig('analysis_results/formula_length_distribution.png')
    plt.close()
    
    # Top elements bar chart
    plt.figure(figsize=(12, 6))
    elements, counts = zip(*composition_stats['top_elements'])
    plt.bar(elements, counts)
    plt.xlabel('Element')
    plt.ylabel('Frequency')
    plt.title('Top 10 Elements in Compositions')
    plt.savefig('analysis_results/top_elements.png')
    plt.close()
    
    # Notation styles bar chart
    plt.figure(figsize=(12, 6))
    styles = list(notation_stats.keys())
    counts = list(notation_stats.values())
    plt.bar(styles, counts)
    plt.xlabel('Notation Style')
    plt.ylabel('Count')
    plt.title('Notation Styles in Compositions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis_results/notation_styles.png')
    plt.close()
    
    # Paper types pie chart
    plt.figure(figsize=(10, 8))
    labels = list(paper_types.keys())
    sizes = list(paper_types.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Paper Type Distribution')
    plt.savefig('analysis_results/paper_types.png')
    plt.close()
    
    # Generate summary report
    with open('analysis_results/test_set_summary.txt', 'w') as f:
        f.write("# Test Set Analysis Summary\n\n")
        
        f.write("## Dataset Statistics\n")
        f.write(f"- Total papers: {len(test_data)}\n")
        f.write(f"- Total compositions: {composition_stats['total_compositions']}\n")
        f.write(f"- Unique compositions: {composition_stats['unique_compositions']}\n")
        f.write(f"- Average compositions per paper: {composition_stats['avg_compositions_per_paper']:.2f}\n")
        f.write(f"- Median compositions per paper: {composition_stats['median_compositions_per_paper']:.2f}\n")
        f.write(f"- Maximum compositions in a paper: {composition_stats['max_compositions_per_paper']}\n\n")
        
        f.write("## Formula Characteristics\n")
        f.write(f"- Average formula length: {composition_stats['formula_length_stats']['mean']:.2f} characters\n")
        f.write(f"- Median formula length: {composition_stats['formula_length_stats']['median']} characters\n")
        f.write(f"- Shortest formula: {composition_stats['formula_length_stats']['min']} characters\n")
        f.write(f"- Longest formula: {composition_stats['formula_length_stats']['max']} characters\n\n")
        
        f.write("## Top 10 Elements\n")
        for element, count in composition_stats['top_elements']:
            f.write(f"- {element}: {count}\n")
        f.write("\n")
        
        f.write("## Notation Styles\n")
        for style, count in notation_stats.items():
            f.write(f"- {style}: {count} compositions\n")
        f.write("\n")
        
        f.write("## Paper Type Distribution\n")
        for paper_type, count in paper_types.items():
            f.write(f"- {paper_type}: {count} papers\n")
    
    print("Analysis complete! Results saved to 'analysis_results' directory.")

if __name__ == "__main__":
    analyze_test_set()
