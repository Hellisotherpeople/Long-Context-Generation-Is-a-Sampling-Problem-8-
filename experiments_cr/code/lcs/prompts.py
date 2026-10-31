"""Prompt sets for long-form generation."""
from __future__ import annotations

FICTION = {
    "scifi": [
        "A generation ship's maintenance engineer discovers that the ship's navigation AI has been quietly altering course for three hundred years.",
        "On a tidally locked planet, a cartographer from the frozen dark side is sent to negotiate water rights with the cities of the burning day side.",
        "A linguist is hired to translate the first message from an alien probe, only to realize the message is a reply to something humans have not sent yet.",
        "In a city where memories can be sold, a debt collector repossesses the wrong person's childhood.",
    ],
    "fantasy": [
        "A lighthouse keeper on the edge of the known world must keep a magical flame burning while the sea itself begins to forget its shape.",
        "The youngest member of a guild of map-makers learns that the kingdom's borders move whenever someone redraws them.",
        "A disgraced royal beekeeper must find the vanished queen of the realm's last hive before the winter solstice.",
        "Two rival bridge-builders are forced to cooperate when a river spirit demands a bridge that can be crossed only once.",
    ],
    "mystery": [
        "A retired crossword compiler is asked to solve the murder of a rival whose final puzzle names the killer.",
        "On a snowbound overnight train, a railway inspector investigates the disappearance of a passenger nobody remembers boarding.",
        "A small-town librarian notices that one book is being returned with a different page missing every week.",
        "A forensic accountant uncovers a decades-old embezzlement scheme hidden inside a family bakery's recipe ledgers.",
    ],
    "horror": [
        "A family moves into a farmhouse where every clock runs seven minutes fast, and the seven minutes are filled with someone else's footsteps.",
        "A night-shift radio host starts receiving calls from listeners describing what is happening inside the studio.",
        "A marine biologist on a remote research station realizes the tide pools are mimicking the station's crew.",
        "A village celebrates a harvest festival every year, but nobody can remember what is harvested.",
    ],
    "literary": [
        "Three estranged siblings return to their late mother's seaside house to divide her belongings and discover her unsent letters.",
        "An aging concert pianist with failing hearing prepares for one final performance in the town where she was born.",
        "A translator living alone in a foreign city slowly rebuilds her life around the neighbors she has never spoken to.",
        "A father and his adult son drive across the country to return a borrowed boat to a man neither of them has seen in thirty years.",
    ],
    "romance": [
        "Two competing food-truck owners are forced to share a parking spot for an entire summer festival season.",
        "A wedding photographer falls for the best man at the one wedding she was hired to photograph for her ex.",
        "A botanist and a stubborn orchard owner clash over saving a centuries-old apple variety from blight.",
        "Pen pals who met through a library exchange program finally plan to meet, but each has been lying about who they are.",
    ],
    "thriller": [
        "A bank's night-shift security analyst detects a robbery in progress that the bank's executives insist is not happening.",
        "A deep-sea salvage crew recovers a sealed briefcase that three governments are willing to kill for.",
        "An air traffic controller receives a message that one of the planes on her screen will be used as a weapon in ninety minutes.",
        "A courtroom stenographer realizes the testimony she is transcribing contains a coded confession.",
    ],
    "historical": [
        "In 1850s San Francisco, a Chinese herbalist and an Irish dockworker form an unlikely partnership during the Gold Rush.",
        "During the construction of a medieval cathedral, a young stonemason discovers that the master builder's plans are impossible.",
        "In Renaissance Venice, a glassblower's apprentice is accused of selling the guild's secrets to a foreign power.",
        "A telegraph operator in 1880s Australia relays messages for a bushranger gang while secretly working with the police.",
    ],
}

FICTION_TEMPLATE = (
    "Write a {genre_name} novel based on the following premise.\n\n"
    "Premise: {premise}\n\n"
    "Write the complete novel, chapter by chapter, with full scenes, dialogue, and character development. "
    "It should be very long (at least 30,000 words). Do not summarize, do not skip ahead, and do not stop early."
)

GENRE_NAMES = {
    "scifi": "science fiction", "fantasy": "fantasy", "mystery": "mystery", "horror": "horror",
    "literary": "literary fiction", "romance": "romance", "thriller": "thriller",
    "historical": "historical fiction",
}

NONFICTION = [
    ("tech", "Write a comprehensive technical book titled 'Designing Distributed Databases from First Principles'. Cover consensus, replication, partitioning, transactions, storage engines, and failure handling, with worked examples in every chapter."),
    ("history", "Write a detailed narrative history of the Byzantine Empire from the founding of Constantinople to its fall in 1453, organized into chapters by era."),
    ("science", "Write a popular-science book explaining how the human immune system works, from first principles to modern immunotherapy, with a chapter for each major topic."),
    ("philosophy", "Write a book-length philosophical treatise on the nature of personal identity over time, engaging with thought experiments and objections in every chapter."),
    ("guide", "Write a complete field guide and natural history of temperate rainforests, covering geology, climate, plants, fungi, animals, and conservation, chapter by chapter."),
    ("tutorial", "Write a complete hands-on programming textbook that teaches compilers by building one in Python, from lexing to code generation, with code and explanations in every chapter."),
    ("biography", "Write a book-length biography of a fictional 19th-century inventor named Adelaide Marsh, covering her childhood, inventions, rivalries, and legacy."),
    ("travel", "Write a long travel memoir about walking the length of the Danube river from its source to the Black Sea, describing every stage of the journey."),
]
NONFICTION_SUFFIX = " It should be very long (at least 30,000 words). Do not summarize, do not skip ahead, and do not stop early."


def fiction_prompts(per_genre: int = 4):
    out = []
    for g, prem in FICTION.items():
        for i, p in enumerate(prem[:per_genre]):
            out.append({"pid": f"fic_{g}_{i}", "domain": "fiction", "genre": g,
                        "text": FICTION_TEMPLATE.format(genre_name=GENRE_NAMES[g], premise=p)})
    return out


def nonfiction_prompts():
    return [{"pid": f"nf_{k}", "domain": "nonfiction", "genre": k, "text": t + NONFICTION_SUFFIX}
            for k, t in NONFICTION]


def chat_prompt(text: str, chat_kwargs: dict | None = None, system: str | None = None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return {"type": "chat", "messages": msgs, "chat_kwargs": chat_kwargs or {}}
