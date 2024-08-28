0. **PREPARE:**
   - **Create virtual environment:**
     ```bash
     virtualenv --python=3.10 env-3.10
     ```
   - **Activate the environment:**
     ```bash
     source env-3.10/bin/activate
     ```
   - **Install the requirements:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Install the nltk packages:**
     ```python
     import nltk
     nltk.download('wordnet')
     nltk.download('stopwords')
     ```

5. **Test the API:**
   ```bash
   curl -X POST "http://localhost:8000/process_text" -H "Content-Type: application/json" -d '{"text": "i am good."}'
   ```

6. **Run the API:**
   ```bash
   uvicorn src.api.api:app --reload
   ```

7. **Run the GUI:**
   ```bash
   streamlit run src/gui/gui.py
   ```
