import re
import ast
import torch
import numpy as np
from itertools import product
from transformers import T5Tokenizer, T5ForConditionalGeneration


# ---------------------------
#  Regex to extract x, y, z
#  (as in original ipynb)
# ---------------------------
def regex_pattern_1(text):
    """
    Extract candidate values of x, y, z from text using his approach.
    Returns a list: [x_values, y_values, z_values].
    """
    pattern_x = r'x\s*=\s*([\d.]+(?:[,;\s*and-–]*[\d.]+)*\b)'
    pattern_y = r'y\s*=\s*([\d.]+(?:[,;\s*and-–]*[\d.]+)*\b)'
    pattern_z = r'z\s*=\s*([\d.]+(?:[,;\s*and-–]*[\d.]+)*\b)'

    def extract_vals(pat, txt):
        found = re.findall(pat, txt, re.IGNORECASE)
        vals = []
        for match in found:
            # replace potential delimiters with commas
            match_clean = match.replace('–', ',').replace(';', ',').replace('and', ',')
            splitted = [v.strip() for v in match_clean.split(',') if v.strip()]
            # keep only numeric
            for s in splitted:
                if s.replace('.', '', 1).isdigit():
                    vals.append(s)
        return vals

    x_vals = extract_vals(pattern_x, text)
    y_vals = extract_vals(pattern_y, text)
    z_vals = extract_vals(pattern_z, text)

    # Filter out nonsense
    x_vals = [v for v in x_vals if 0 < float(v) < 100]
    y_vals = [v for v in y_vals if 0 < float(v) < 100]
    z_vals = [v for v in z_vals if 0 < float(v) < 100]

    return [list(set(x_vals)), list(set(y_vals)), list(set(z_vals))]


# ---------------------------
#  Safely parse T5 JSON-like
#  output into Python object
# ---------------------------
def safe_literal_eval(txt):
    """Safely parse a string to Python object, returning [] on failure."""
    try:
        return ast.literal_eval(txt)
    except:
        return []



import ast
import operator as op

# Map each AST operator node to the corresponding Python function
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv
}
def eval_(node):
    if isinstance(node, ast.Num):
        # For literal numbers
        return node.n
    elif isinstance(node, ast.BinOp):
        # For binary operations, recursively evaluate the left and right nodes
        # then apply the operator from the 'operators' map
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    else:
        # If we encounter anything other than a Number or BinOp, raise an error
        raise TypeError(node)


# ---------------------------
#  Evaluate an arithmetic
#  expression like (x)/(1-x)
# ---------------------------
def eval_expr(expr):
    """
    Evaluate a simple arithmetic expression safely (only +, -, *, /).
    E.g. '(x)/(1-x)'.
    """
    # If you want to replicate exactly the AST approach from the ipynb,
    # you can do so. For brevity, we do a simple Python eval:
    # 1. Parse the string expression into an AST
    # 2. Extract the AST's body
    # 3. Recursively evaluate only safe arithmetic operations
    return eval_(ast.parse(expr, mode='eval').body)



# ---------------------------
#  Substitute x,y,z combos
#  & normalize to sum=100
#  (As in "substituteRegex")
# ---------------------------
def substitute_xyz(parsed_comps, x_vals, y_vals, z_vals):
    """
    parsed_comps could be e.g. [[('SiO2','(x)/(x+1-x)'), ... ], [...]] 
    We'll do cartesian product of x_vals, y_vals, z_vals, and evaluate each expression.
    """
    if not parsed_comps:
        return []

    # Usually the model outputs a list-of-lists. We handle only the first if multiple:
    # e.g. [ [ ('SiO2','(x)/(x+...'), ... ] ]
    composition_block = parsed_comps[0]

    # If no x,y,z found, treat them as [None] so it doesn't blow up
    if not x_vals: x_vals = [None]
    if not y_vals: y_vals = [None]
    if not z_vals: z_vals = [None]

    final_list = []
    for (xx, yy, zz) in product(x_vals, y_vals, z_vals):
        new_comp = []
        for (cname, expr_str) in composition_block:
            if isinstance(expr_str, (int, float)):
                # already numeric
                new_comp.append((cname, float(expr_str)))
            else:
                # do textual substitution
                tmp = str(expr_str)
                if xx is not None:
                    tmp = tmp.replace('x', xx)
                if yy is not None:
                    tmp = tmp.replace('y', yy)
                if zz is not None:
                    tmp = tmp.replace('z', zz)
                try:
                    val = eval_expr(tmp)
                    new_comp.append((cname, float(val)))
                except:
                    new_comp.append((cname, 0.0))

        # Now normalize so sum=100, as in your ipynb
        sm = sum([x[1] for x in new_comp])
        if sm > 0:
            normalized = [(c[0], (c[1]/sm)*100.0) for c in new_comp]
            final_list.append(normalized)
    return final_list


class DhruvilPrompter:
    """
    Pipeline to replicate the author's code and hyperparams:

    1) comp_classifier: composition vs non-composition
       - max_input_length=700, model.generate(..., max_new_tokens=10, num_beams=2)
    2) type_classifier: direct vs equational
       - same hyperparams as comp_classifier
    3) direct extractor:
       - max_input_length=300, model.generate(..., max_new_tokens=800, num_beams=1)
    4) equational extractor:
       - same hyperparams as direct extractor
       - then do x,y,z substitution
    """

    def __init__(
        self,
        composition_classifier_path,
        direct_equational_classifier_path,
        direct_extractor_path,
        equational_extractor_path,
        device=None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ----------- 1) Composition vs Non-composition -----------
        self.comp_tokenizer = T5Tokenizer.from_pretrained(composition_classifier_path)
        self.comp_model = T5ForConditionalGeneration.from_pretrained(composition_classifier_path)
        self.comp_model.to(self.device)

        # ----------- 2) Direct vs Equational -----------
        self.type_tokenizer = T5Tokenizer.from_pretrained(direct_equational_classifier_path)
        self.type_model = T5ForConditionalGeneration.from_pretrained(direct_equational_classifier_path)
        self.type_model.to(self.device)

        # ----------- 3) Direct extractor -----------
        self.direct_tokenizer = T5Tokenizer.from_pretrained(direct_extractor_path)
        self.direct_model = T5ForConditionalGeneration.from_pretrained(direct_extractor_path)
        self.direct_model.to(self.device)

        # ----------- 4) Equational extractor -----------
        self.eqn_tokenizer = T5Tokenizer.from_pretrained(equational_extractor_path)
        self.eqn_model = T5ForConditionalGeneration.from_pretrained(equational_extractor_path)
        self.eqn_model.to(self.device)

        # For replicating the original code’s input lengths:
        self.classifier_max_input_length = 700
        self.extractor_max_input_length = 300

    def _run_classifier(self, tokenizer, model, text):
        """
        Classifier generation with EXACT hyperparams from ipynb:
          max_input_length=700
          generate(..., max_new_tokens=10, num_beams=2, temperature=0.0)
        """
        inputs = tokenizer(
            text,
            max_length=self.classifier_max_input_length,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10,   # EXACT from ipynb
                num_beams=2,         # EXACT from ipynb
                temperature=0.0
            )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _run_extractor(self, tokenizer, model, text):
        """
        Extraction generation with EXACT hyperparams from ipynb:
          max_input_length=300
          generate(..., max_new_tokens=800, num_beams=1, temperature=0.0)
        """
        inputs = tokenizer(
            text,
            max_length=self.extractor_max_input_length,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=800,   # EXACT from ipynb
                num_beams=1,          # EXACT from ipynb
                temperature=0.0
            )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def classify_comp(self, text):
        """
        Step 1: Composition vs Non-composition
        Returns "0" or "1".
        """
        return self._run_classifier(self.comp_tokenizer, self.comp_model, text).strip()

    def classify_direct_eqn(self, text):
        """
        Step 2: Direct vs Equational
        Returns "0" or "1", or possibly "direct"/"equational" depending on your training labels.
        """
        return self._run_classifier(self.type_tokenizer, self.type_model, text).strip()

    def extract_direct(self, text):
        """
        Direct composition extraction
        Returns a Python list-of-lists-of-tuples.
        """
        raw_str = self._run_extractor(self.direct_tokenizer, self.direct_model, text)
        return safe_literal_eval(raw_str)

    def extract_equational(self, text):
        """
        Equational composition extraction (raw param form).
        Returns list-of-lists-of-tuples with x,y,z placeholders.
        """
        raw_str = self._run_extractor(self.eqn_tokenizer, self.eqn_model, text)
        return safe_literal_eval(raw_str)

    def __call__(self, text):
        """
        Full pipeline:
          1) classify comp vs non-comp
          2) if comp => classify direct/eqn
          3) if direct => extract
          4) if eqn => extract + substitute
        """
        # Step 1: comp vs non-comp
        is_comp = self.classify_comp(text)
        if is_comp == "0":
            # Non-composition
            return []

        # Step 2: direct vs eqn
        classification = self.classify_direct_eqn(text)
        # Depending on your labels, might be "direct" or "1" etc.
        # Suppose "1" => direct, "0" => eqn (or "direct" => direct, etc.)
        if classification.lower() == "1":
            # Direct
            comps = self.extract_direct(text)
            return comps
        else:
            # Equational
            eqn_comps = self.extract_equational(text)
            # parse x,y,z from text
            x_vals, y_vals, z_vals = regex_pattern_1(text)
            # do final numeric substitution
            final_result = substitute_xyz(eqn_comps, x_vals, y_vals, z_vals)
            return final_result


# ------------------ Example usage ------------------
if __name__ == "__main__":
    # Provide your local model paths
    comp_classifier_path = "mtp_trainClassifierWithout100_ratio1to6_run1_FlanT5Large.pt"
    direct_eqn_classifier_path = "mtp_trainClassifierWithout100_dm_vs_eqn_run1_FlanT5Large.pt"
    direct_extractor_path = "mtp_CompExtractor_Without100_FlanT5Large_DirectMatch_AdditionalGpt4Data_run_2.pt"
    eqn_extractor_path = "mtp_CompExtractor_Without100_FlanT5Large_OnlyEqn_AdditionalGpt4Data_run_2.pt"

    pipeline = DhruvilPrompter(
        comp_classifier_path,
        direct_eqn_classifier_path,
        direct_extractor_path,
        eqn_extractor_path
    )

    test_sentence = "The glass 20Li2O–80B2O3 was prepared; x=0.3"
    result = pipeline(test_sentence)
    print("Extraction =>", result)
