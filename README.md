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

1. **DATA:** [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz)

2. **Create the dataset from DATA:**
   ```bash
   python src/data/make_dataset.py --input_path ./data/raw/review_polarity/txt_sentoken/ --output_file data/processed/data.txt
   ```

3. **Train the model:**
   ```bash
   python src/models/train_model.py --train_file data/processed/data.txt --model_type svm --output_model models/model.pkl
   ```

4. **Test the model:**
   ```bash
   python src/models/predict_model.py --input_sentence 'I am good' --model_file models/model.pkl
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
