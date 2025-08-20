# 📄 Contract Analysis System - Unfair Clause Detection

An advanced contract analysis system that uses **LexGLUE methodology** and state-of-the-art NLP models to detect potentially unfair clauses in contracts.

## 📸 App Preview

![Contract Analysis App](images/screenshot.png)

*🚀 Live demo: [Contract Analysis App](https://legalaideepresearch-rzy5kk9k2ygxudlavpsg4z.streamlit.app)*

## 🎯 **Primary Features**

### Unfair Clause Detection
- **Pattern-based Detection**: Rule-based identification of known unfair clause patterns
- **ML-based Detection**: Fine-tuned BERT models for intelligent clause classification  
- **LexGLUE Integration**: Uses UNFAIR-ToS models specifically trained for legal text
- **Multi-format Support**: PDF, DOCX, DOC, and TXT file processing

### Advanced Analysis
- **Risk Assessment**: Categorizes clauses by unfairness severity
- **Detailed Recommendations**: Provides specific suggestions for contract improvements
- **Export Capabilities**: Download analysis results in multiple formats
- **Interactive Interface**: User-friendly Streamlit web application

## 🚀 **Quick Start**

### Online Demo
Try the live demo: [Contract Analysis App](https://legalaideepresearch-rzy5kk9k2ygxudlavpsg4z.streamlit.app)

### Local Installation

1. **Clone the repository**
   \`\`\`bash
   git clone https://github.com/leonjas/legal_ai_deep_research.git
   cd legal_ai_deep_research
   \`\`\`

2. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Download spaCy model**
   \`\`\`bash
   python -m spacy download en_core_web_sm
   \`\`\`

4. **Run the application**
   \`\`\`bash
   streamlit run app/main.py
   \`\`\`

## 📋 **Supported File Formats**

- **PDF** - Extract text from PDF contracts
- **DOCX/DOC** - Microsoft Word documents
- **TXT** - Plain text files
- **Direct Input** - Paste contract text directly

## 🛠 **Technology Stack**

- **Frontend**: Streamlit
- **ML Models**: Transformers, PyTorch, spaCy
- **Document Processing**: PyMuPDF, python-docx
- **Visualization**: Matplotlib, Plotly, Seaborn

## 📊 **Analysis Methods**

1. **Pattern-based Detection**: Uses predefined rules to identify common unfair clause patterns
2. **Machine Learning Classification**: Employs fine-tuned BERT models for intelligent analysis
3. **Hybrid Approach**: Combines both methods for comprehensive coverage

## 🎛 **Configuration**

- **Confidence Threshold**: Adjustable sensitivity for unfair clause detection
- **Analysis Method**: Choose between Pattern-based, ML-based, or Both
- **Export Options**: Multiple output formats for analysis results

## 📄 **License**

This project is licensed under the MIT License.

## 👨‍💻 **Author**

Created by [LiangPoYen](https://github.com/leonjas) - jasmineyen16008@gmail.com

## 🤝 **Contributing**

Contributions, issues, and feature requests are welcome!

## ⭐ **Support**

If you find this project helpful, please give it a star on GitHub!
