# CS 224W Final Project

## Personalized Recommendation System for Questions to Answer on SuperUser

This is a class project for CS 224W (Analysis of Networks) at Stanford.

[Final Report](https://github.com/gkovacs/cs224w/raw/master/final-report.pdf)

[Final Presentation](cs224w-final-presentation.pdf)

## Abstract

We have built a personalized recommendation engine that is able to predict, at a given snapshot of time, the probability of a given user answering a particular SuperUser question. We use this in the application of generating personalized suggestions of questions that each user would be interested in answering. We have developed 2 components to recommend relevant questions to users: one which computes the match in a question’s topic to a potential answerer’s interests, and another which predicts the likelihood of an answer coming in for a question at a given point in time. We combine these results by multiplying and normalizing to generate an overall estimate of the probability a question is answered by a given user at a given time. We use community detection at the tag-similarity level to establish recommended questions for particular communities, and use this to diversity recommendations, particularly for users who have limited history on the site. Results show that our combined approach is more accurate at modeling question-answering events on a subset of the StackOverflow question-answering network than either of the individual components alone. Our final model is able to generate recommendations such that the actual question answered is among the 29 top recommendations.

## License

MIT

## Authors

[Geza Kovacs](https://www.gkovacs.com/)

Arpad Kovacs

Shahriyar Pruisken

