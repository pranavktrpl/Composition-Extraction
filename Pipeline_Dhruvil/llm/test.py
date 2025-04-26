from llamat_prompter import llamatPrompter

def test_llamat_prompter(test_prompt:str):
    # Initialize the prompter
    prompter = llamatPrompter()
    
    # Test prompt
    # test_prompt = "Hi! Just trying to check are you working?"
    
    # Create the formatted prompt
    prompt = (
        f"<|im_start|>\n{test_prompt}\n"
    )
    
    # Get the response
    response = prompter(prompt)
    
    # Print the results
    print("Test Prompt:", test_prompt)
    print("\nResponse:", response)

if __name__ == "__main__":
    Test_prompt = ""
    # with open("prompts/llamat_complete_prompt.txt", "r") as f:
    #     Test_prompt = f.read()

    context = ""
    # context = "MoO3 thin films were prepared, in this work, evaporating the material with a CO2 laser set up in continuous wave mode. The films were deposited on glass substrates at different substrate temperatures and high vacuum. Additionally, other samples were prepared at different oxygen pressures in the chamber. The samples were characterized by X-ray photoelectron spectroscopy (XPS) and optical transmittance measurements in the visible range. The XPS analyses showed evidence of thin film formation with two composition mix: MoO3 and MoO2; this result appeared in all fabricated samples. The transmittance spectra were reproduced theoretically and from the simulation were calculated the index of refraction, the absorption coefficient of the material, the forbidden energy gap and sample thickness. Samples prepared at high substrate temperature presented, in the red region, an absorption band of about 2.5×104 cm−1. This band kept its wavelength position when oxygen was injected in the chamber but its height decreased by one order of magnitude, indicating it is associated to oxygen vacancies in the sample. \n "
    with open("prompting_data/research-paper-text/S0167577X05011511.txt", "r") as f:
        context = context + f.read()

    with open("prompting_data/research-paper-tables/S0040609004014671.txt", "r") as f:
        context = context + f.read()

    instruction = f"You are a material scientist assisting me link compositions with properties in a research paper. Use the below research paper as context - {context}. Now based on the above context, from this list of compositions = [MoO2, MoO2, CO2, GaAs, Cu, Al, OH, H2O,], \n\n question: what chemical composition is mentioned to have a refractive index of '1.71' in table-1 of the table?\n\n <|im_end|> <|im_start|> answer: "

    # incomplete_table = ""
    # with open("prompting_data/Matskraft-tables/S0167577X05011511.txt", "r") as f:
    #     incomplete_table = f.read()

    # Test_prompt = Test_prompt.replace("{{Research Paper}}", context)
    # Test_prompt = Test_prompt.replace("{{Table}}", incomplete_table)

    print(instruction)
    test_llamat_prompter(instruction)
