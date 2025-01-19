# NLP_GroupProject
Technical Report: PlateMate – Your Dining Companion
The application is a question answering system. Two versions of the application were created:
## Architecture V1 (Gpt-4o-mini with Azure AI): 
Relatively expensive, yet very accurate solution. Easily implemented through Azure. Model improvement is highly dependent on future developments of AzureAI.
File Compatibility: Supports .pdf, .doc, and .xlsx formats, ensuring compatibility with the standard restaurant menu formats.
Multilingual and Symbol Recognition: Accurately processes menus in multiple languages and recognizes allergen symbols, aligning with industry standards.
Results were improved using prompt engineering.
Performance could be improved through more powerful, albeit expensive, models. Gpt-4o-mini was selected for its superior price/performance ratio. 

## Architecture V2 (Multi-Model Architecture): 
A cheaper, albeit more complex architecture. It is less accurate due to errors introduced by misclassification and the less powerful simpleQA model. Results will be improved as more data is gathered to finetune the application.
DeBERTa Zero-Shot Classifier: Classifies user queries as menu-related or general Q&A questions. 
Gpt-4o-mini with Azure AI: Handles complex menu-related queries for high-quality answers. 
SimpleQA with BERT: answers straight-forward Q&A provided by the restaurants. 
Challenges and Solutions
Classifier accuracy
Different combinations of models, labels and context were tested, leading to an outstanding performance of nearly 80% of accuracy. 
## Deployment Issues:
Due to large model sizes, exporting models via pickle was infeasible. Caching models significantly reduced response times from 3 minutes to approximately 10 seconds.
Dependency conflicts prevented deployment of Architecture V.2 on Streamlit Cloud. As an alternative, the application can be run in Google Colab.
## Recommendations for Future Iterations
- Implement and fine tune a better model to answer simple and straightforward Q&A.
- Conduct broader paraphrasing and real-world scenario testing to ensure robustness in user interactions. 
- Include a translator for V2: As this is one of the most valuable features that’s already included in the V1. 
- Resolve dependency issues for cloud-based deployment on platforms like Streamlit or Azure.

