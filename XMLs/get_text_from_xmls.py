import os
import argparse
from lxml import etree
from tqdm import tqdm
import re  # Import the regex module

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract text from XML files and save as TXT.")
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to the input directory containing XML files.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output directory where TXT files will be saved.'
    )
    return parser.parse_args()

def extract_text(element):
    """
    Recursively extract text from an XML element, ignoring tags but preserving the text content.
    Cleans up multiple spaces and line breaks while preserving essential formatting.
    """
    text = ''.join(element.itertext()).strip()
    # Replace multiple whitespace characters (including line breaks) with a single space
    clean_text = re.sub(r'\s+', ' ', text)
    # Optionally, handle specific patterns if necessary
    # Example: Remove space before and after certain symbols
    clean_text = re.sub(r'\s*([=±∼])\s*', r'\1', clean_text)
    return clean_text

def get_all_namespaces(xml_path):
    """
    Extract all namespaces from the XML file.
    """
    namespaces = dict([
        node for _, node in etree.iterparse(xml_path, events=['start-ns'])
    ])
    return namespaces

def get_contents(xml_path, exclude_sections=None):
    if exclude_sections is None:
        exclude_sections = []
    try:
        # Parse XML and extract namespaces
        tree = etree.parse(xml_path)
        root = tree.getroot()
        namespaces = get_all_namespaces(xml_path)

        paper = {}

        # Helper function to find elements regardless of namespace
        def find_element(xpath_expr):
            return root.find(xpath_expr, namespaces=namespaces)

        def find_all_elements(xpath_expr):
            return root.findall(xpath_expr, namespaces=namespaces)

        # Extract Title
        title_elem = find_element('.//dc:title') or find_element('.//ce:title') or find_element('.//title')
        paper['Title'] = ' '.join(title_elem.text.split(',')).strip() if title_elem is not None and title_elem.text else ''

        # Extract Abstract
        abstract_elem = find_element('.//ce:abstract') or find_element('.//dc:description') or find_element('.//abstract')
        if abstract_elem is not None:
            abstract_text = extract_text(abstract_elem).replace('Abstract', '').strip()
        else:
            abstract_text = ''
        paper['Abstract'] = abstract_text

        # Extract Keywords
        keywords = find_all_elements('.//ce:keywords/ce:keyword') or find_all_elements('.//keywords/keyword') or find_all_elements('.//keyword')
        keyword_list = []
        for kw in keywords:
            kw_text = extract_text(kw)
            if kw_text:
                keyword_list.append(kw_text)
        if keyword_list:
            paper['Keywords'] = ', '.join(keyword_list)

        # Extract Acknowledgements
        acknowledgments = find_all_elements('.//ce:acknowledgment') or find_all_elements('.//acknowledgment') or find_all_elements('.//acknowledgements')
        if acknowledgments:
            ack_texts = []
            for ack in acknowledgments:
                ack_title_elem = ack.find('.//ce:section-title', namespaces=namespaces) or ack.find('.//section-title', namespaces=namespaces)
                ack_title = extract_text(ack_title_elem) if ack_title_elem is not None else 'Acknowledgements'
                ack_content = extract_text(ack)
                if ack_title and ack_content:
                    ack_texts.append(f"{ack_title}:\n{ack_content}")
                elif ack_content:
                    ack_texts.append(f"Acknowledgements:\n{ack_content}")
            if ack_texts:
                paper['Acknowledgements'] = '\n\n'.join(ack_texts)

        # **Removed: Extract References**
        # The following block has been removed to prevent the extraction of the References section.
        # references = find_all_elements('.//ce:bibliography') or find_all_elements('.//bibliography') or find_all_elements('.//references')
        # if references:
        #     ref_texts = []
        #     for ref in references:
        #         ref_title_elem = ref.find('.//ce:section-title', namespaces=namespaces) or ref.find('.//section-title', namespaces=namespaces)
        #         ref_title = extract_text(ref_title_elem) if ref_title_elem is not None else 'References'
        #         ref_content = extract_text(ref)
        #         if ref_title and ref_content:
        #             ref_texts.append(f"{ref_title}:\n{ref_content}")
        #         elif ref_content:
        #             ref_texts.append(f"References:\n{ref_content}")
        #     if ref_texts:
        #         paper['References'] = '\n\n'.join(ref_texts)

        # Define sections to exclude (case-insensitive)
        exclusion_set = set([sec.lower() for sec in exclude_sections])

        # Extract all other sections dynamically, excluding specified sections
        sections = find_all_elements('.//ce:section-title') or find_all_elements('.//section-title')
        for sec in sections:
            section_title = extract_text(sec).lower()
            if section_title in exclusion_set:
                # Skip excluded sections
                continue
            # Get the parent element which contains the content
            parent = sec.getparent()
            if parent is not None:
                # Assume that the content is in sibling elements after the section title
                content = []
                start_collecting = False
                for sibling in parent.iterchildren():
                    if sibling is sec:
                        start_collecting = True
                        continue
                    if start_collecting:
                        # Stop if another section title is encountered
                        if sibling.tag.endswith('section-title'):
                            break
                        # Extract text from the sibling
                        text = extract_text(sibling)
                        if text:
                            content.append(text)
                if content:
                    # Capitalize the section title for consistency
                    capitalized_title = ' '.join([word.capitalize() for word in section_title.split()])
                    paper[capitalized_title] = '\n'.join(content)

        # Extract Body Sections (like paragraphs not under specific section titles)
        body_sections = find_all_elements('.//ce:sections/ce:para') or find_all_elements('.//sections/para') or find_all_elements('.//para')
        if body_sections:
            body_text = '\n\n'.join([extract_text(para) for para in body_sections if extract_text(para)])
            paper['Body'] = body_text

        # Combine all sections into a single string
        extracted_text = ""
        for key, value in paper.items():
            extracted_text += f"{key}:\n{value}\n\n"

        return extracted_text.strip()
    except etree.XMLSyntaxError as e:
        raise ValueError(f"XML Syntax Error: {e}")
    except Exception as e:
        raise ValueError(f"Error processing XML: {e}")
    

def main():
    args = parse_arguments()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all XML files in the input directory
    xml_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.xml')]

    if not xml_files:
        print(f"No XML files found in the input directory '{input_dir}'.")
        return

    # Define sections to exclude
    sections_to_exclude = ['references', 'bibliography', 'ref', 'bibliographies', 'reference']

    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        xml_path = os.path.join(input_dir, xml_file)
        try:
            extracted_text = get_contents(xml_path, exclude_sections=sections_to_exclude)
            txt_filename = os.path.splitext(xml_file)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(extracted_text)
        except ValueError as ve:
            print(f"Failed to process '{xml_file}': {ve}")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{xml_file}': {e}")

    print(f"Extraction completed. TXT files are saved in '{output_dir}'.")

if __name__ == "__main__":
    main()
