import re

html_path = r"c:\Temp\11-12 SE\12SE\12SE_HSC_subject_selection\UI_test\survey_train.html"

with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

select_template = """<select id="target_subject_{i}" name="target_subject_{i}" {req}>
            <option value="" disabled selected>Select a subject...</option>
            <optgroup label="English">
              <option value="English Advanced">English Advanced</option>
              <option value="English Extension">English Extension</option>
              <option value="English Standard">English Standard</option>
            </optgroup>
            <optgroup label="HSIE">
              <option value="Ancient History">Ancient History</option>
              <option value="Biblical Studies">Biblical Studies</option>
              <option value="Business Studies">Business Studies</option>
              <option value="Commerce">Commerce</option>
              <option value="Community and Family Studies">Community and Family Studies</option>
              <option value="Economics">Economics</option>
              <option value="Geography">Geography</option>
              <option value="Legal Studies">Legal Studies</option>
              <option value="School of Languages">School of Languages</option>
              <option value="Society and Culture">Society and Culture</option>
            </optgroup>
            <optgroup label="Maths">
              <option value="Mathematics Advanced">Mathematics Advanced</option>
              <option value="Mathematics Extension 1">Mathematics Extension 1</option>
              <option value="Mathematics Extension 2">Mathematics Extension 2</option>
              <option value="Mathematics Standard">Mathematics Standard</option>
              <option value="Software Engineering">Software Engineering</option>
            </optgroup>
            <optgroup label="PDHPE">
              <option value="Health and Movement Science">Health and Movement Science</option>
            </optgroup>
            <optgroup label="Science">
              <option value="Biology">Biology</option>
              <option value="Physics">Physics</option>
            </optgroup>
            <optgroup label="TAS">
              <option value="Design and Technology">Design and Technology</option>
              <option value="Engineering Studies">Engineering Studies</option>
              <option value="Food Technology">Food Technology</option>
              <option value="Hospitality">Hospitality</option>
              <option value="Industrial Technology Multimedia">Industrial Technology Multimedia</option>
              <option value="Industrial Technology Timber">Industrial Technology Timber</option>
            </optgroup>
            <optgroup label="Visual and Performing Arts">
              <option value="Drama">Drama</option>
              <option value="Music">Music</option>
              <option value="Visual Arts">Visual Arts</option>
            </optgroup>
          </select>"""

for i in range(1, 8):
    # Regex to find the <select> tag and all its inner content until </select>
    pattern = rf'( *)<select id="target_subject_{i}".*?</select>'
    req = 'required' if i < 7 else ''
    
    def replacer(match):
        spaces = match.group(1)
        return spaces + select_template.format(i=i, req=req)
        
    content = re.sub(pattern, replacer, content, flags=re.DOTALL)

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated survey_train.html with new clusters successfully.")
