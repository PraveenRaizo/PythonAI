import nltk
from nltk.draw.tree import draw_trees
from tkinter import Tk

# DT - deterministic, JJ - adjective, NN - Noun, VBP - verb, IN - preposition
sentence = [("a", "DT"), ("clever", "JJ"), ("fox", "NN"), ("was", "VBP"),("jumping", "VBP"), ("over", "IN"), ("the", "DT"), ("wall", "NN")]

grammar = "NP:{<DT>?<JJ>*<NN>}" # grammar given in the form of regular expression

'''
    Multiline comment :
    "NP:" indicates that the pattern is for a noun phrase.
    "{...}" encloses the pattern itself.
    "<DT>?" specifies an optional determiner (DT) tag. The question mark "?" denotes zero or one occurrence of the preceding element.
    "<JJ>" denotes zero or more adjectives (JJ) occurring in the noun phrase. The asterisk "" indicates zero or more occurrences of the preceding element.
    "<NN>" specifies a mandatory noun (NN) tag.
    Putting it all together, "NP:{<DT>?<JJ>*<NN>}" matches noun phrases that consist of an optional determiner, zero or more adjectives, and a mandatory noun.
'''

# defining a parser that will parse the grammar:
parser_chunking = nltk.RegexpParser(grammar)

# parser parses the sentence as follows:
output_chunk = parser_chunking.parse(sentence)

# Draw the parse tree using tkinter
trees = [output_chunk]
draw_trees(*trees)

# Create a Tkinter window and display the parse tree
root = Tk()
canvas = trees[0]
canvas.draw()
root.mainloop()