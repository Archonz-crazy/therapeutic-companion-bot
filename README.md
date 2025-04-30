# Therapeutic Companion Bot

A chatbot designed to provide conversational support and resources for mental health, leveraging advanced Natural Language Processing (NLP) and curated datasets from trusted sources.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Data Collection](#data-collection)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Resume Pointers](#resume-pointers)

---

## Introduction

**Therapeutic Companion Bot** is an open-source chatbot project focused on mental health support. It utilizes NLP techniques to engage users in meaningful conversations, answer questions, and provide guidance based on data from reputable sources. The bot is designed with ethical considerations in mind, following best practices for responsible AI in mental health contexts.

---

## Features

- Conversational AI for mental health support
- Data-driven responses sourced from real mental health discussions
- Ethical guardrails based on World Health Organization (WHO) guidelines
- Multi-format data ingestion (.csv, .parquet, .txt, .json)
- Scalable architecture for large datasets (5GB+)

---

## Data Collection

The bot's knowledge base is built from:

- **Reddit:** Scraped mental health subreddits, including subreddit names, user questions, and community responses.
- **World Health Organization (WHO):** Ethical guidelines for mental health chatbots, used to implement conversational guardrails.
- **HuggingFace:** Processed Q&A datasets simulating psychiatrist-patient interactions.

All data was collected and processed into multiple formats to ensure flexibility and scalability.

---

## Tech Stack

- **Programming Language:** Python
- **NLP Libraries:** (e.g., HuggingFace Transformers, NLTK, spaCy)
- **Data Handling:** Pandas, NumPy
- **Scraping Tools:** Requests, BeautifulSoup, etc.

---

## Installation

1. Clone the repository:

