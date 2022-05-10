# Exploring Contrastive Learning for Multimodal Detection of Misogynistic Memes

This project aims to apply the idea of contrastive learning for multimodal problem created by the authors of OpenAI's [CLIP](https://github.com/openai/CLIP). The aim is to experiment with contrastive learning to address the detection of misogynistic memes within the context of SemEval-2022 Task 5. Due to its novelty, few solutions to date have explored the use of CLIP in challenging scenarios such as misogynistic memes classification. For this work, we trained a version of CLIP resembling Shariatnia's [application](https://github.com/moein-shariatnia/OpenAI-CLIP.).

This reposiroty contains the project as a Notebook and as a script. This project is a starting point for contrastive learning solutions for multimodal misogynistic meme detection. 

### Arguments for the script

The arguments to use the script

```sh
'-p', type=str, default='dataset', help='Path to dataset'
'-mode',  type=int, default=0, help='0: to test, 1: to train'
'-cleaned', type=int, default=1, help='0: otw, 1: if cleaned_texts.csv exists'
'-processed', type=int, default=1, help='0: otw, 1: if processed_texts.csv exists'
'-split',  type=int, default=0, help='1: Perform the split of dataset, otw: Do not perform split'
'-stat',  type=int, default=0, help='1: Show statistics, otw: Do not show stats'
'-train',  type=str, default="train.csv", help='name of csv file'
'-test',  type=str, default="test.csv", help='name of csv file'
'-valid',  type=str, default="validation.csv", help='name of csv file'
```

Author
----
charicf
