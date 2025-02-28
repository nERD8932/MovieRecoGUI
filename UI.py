import sys
from idlelib.search import SearchDialog
from scipy import spatial
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
import asyncio
from PyQt6.QtCore import Qt, pyqtSignal as Signal, QMimeData
from PyQt6.QtGui import QFont, QDropEvent
from PyQt6.QtWidgets import *
from qasync import QApplication, asyncSlot, QEventLoop  # Correct async integration


def cosineDistance(a, b):
    return spatial.distance.cosine(a, b)

class MainWindow(QMainWindow):
    """
        Here We define a PyQt6 GUI for the Recommendation Engine
    """
    def __init__(self, options: list[str], db: pd.DataFrame):
        super().__init__()

        # Search Variables
        self.searchResults = []
        self.searching = False
        self.searchText = ""
        self.options = options

        # Previously saved Movie Vectors
        self.db = db

        # GUI Definition
        self.setFixedHeight(800)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(20, 50, 20, 50)
        self.setWindowTitle("Movie Recommendation Engine")

        self.apptitle = QLabel("Movie Recommendation Engine")
        self.apptitle.setMargin(20)
        self.apptitle.setFont(QFont("Arial", 25))
        self.apptitle.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.apptitle)

        self.lineEdit = QLineEdit()
        self.lineEdit.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.lineEdit.textChanged.connect(self.Update)
        self.lineEdit.setPlaceholderText("Enter a movie you like.")

        self.resbox = QListWidget(parent=self.lineEdit)
        self.resbox.setDragEnabled(True)
        self.resbox.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.resbox.setMaximumHeight(0)

        l2 = QVBoxLayout()
        search_container = QWidget()
        search_container.setLayout(l2)

        l2.setAlignment(Qt.AlignmentFlag.AlignTop)
        l2.addWidget(self.lineEdit)
        l2.addWidget(self.resbox)
        layout.addWidget(search_container)

        self.draglabel = QLabel("(Drag up to 5 movies into the box below)", parent=self)
        self.draglabel.setStyleSheet(f'color: grey;')
        self.draglabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.draglabel)

        self.likedbox = self.QListWidgetSignal(parent=search_container)
        self.likedbox.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.likedbox.setFixedHeight(108)
        layout.addWidget(self.likedbox)

        self.recobutton = QPushButton("Recommend", parent=self.draglabel)
        self.recobutton.setStyleSheet(f'color: white;')
        self.recobutton.setFixedSize(100, 30)
        self.recobutton.clicked.connect(self.Recommend)

        self.recbox = self.QListWidgetSignal()
        self.recbox.setFixedHeight(108)


        layout.addWidget(self.recobutton, alignment=Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.recbox)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # Extract information from GUI and run recomendation algo
    def Recommend(self):
        self.recbox.clear()
        liked = [self.likedbox.item(i).text() for i in range(self.likedbox.count())]
        if len(liked) > 0:
            vectors = self.db.loc[self.db['MovieName'].isin(liked), 'Vector']
            self.findDistances(vectors)


    @asyncSlot()
    async def findDistances(self, vectors):

        # Get all movie vectors as a dataframe
        vecslice = self.db.loc[:, 'Vector']
        filtered_db = pd.DataFrame(vecslice)

        # Define a column Distances which calculates the average Cosine Distance from all the user
        # selected movie vectors. This will produce a column with a value that denotes how
        # similar or dissimilar a movie is to the user selected movies.
        filtered_db['Distances'] = vecslice.apply(self.avgdistance, args=(vectors,))

        # Sort by that value and pick the top 10 + number of movies selected by user
        filtered_db = filtered_db.sort_values(by=['Distances'], ascending=True)[0:(10 + len(vectors))]

        # Filter out the user selected movies
        filtered_db = filtered_db.loc[~filtered_db['Vector'].isin(vectors)]

        # Join the Movie names for user readability
        filtered_db = filtered_db.join(self.db['MovieName'])

        # Add the movies with the smallest average cosine distance (ie, most similar) to the GUI
        for m in filtered_db['MovieName'].tolist():
            self.recbox.addItem(m)

    # Calculate the average cosine distance
    def avgdistance(self, row, vectors):
        return sum([cosineDistance(row, vectors.iloc[i]) for i in range(len(vectors))])/len(vectors)

    # Search for entered movie if not already searching
    def Update(self, text):
        self.searchText = text.lower()
        if not self.searching:
            self.searching = True
            self.Search(self.searchText)

    # Async method to add and remove GUI elements based on search results
    @asyncSlot()
    async def Search(self, text):
        if len(text) >= 1:
            self.resbox.clear()

            # If search term is in movie name or if movie name is in search term
            self.searchResults = [x for x in self.options if text in x.lower() or x.lower() in text]

            # If such a movie doesn't exist
            if len(self.searchResults) == 0:
                self.resbox.addItem(QListWidgetItem("No results found..."))

            # Add results to GUI
            for result in self.searchResults:
                self.resbox.addItem(QListWidgetItem(result))
            self.resbox.setMaximumHeight((self.resbox.count() + 1) * 14)
            self.searching = False
        else:
            self.resbox.clear()
            # Hide element when inactive
            self.resbox.setMaximumHeight(0)
            self.searching = False

    # Custom class to allow for drag and drop filtering
    class QListWidgetSignal(QListWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        # Overridden
        def dropEvent(self, event: QDropEvent, **kwargs):
            # If user drags in more than 5 elements, remove the first dropped item first
            if self.count() == 5:
                self.takeItem(0)
            super().dropEvent(event)
            items = [self.item(i).text() for i in range(self.count())]
            if len([x for x in items if x == items[-1]]) >1 or items[-1] == "No results found...":
                 self.takeItem(self.count() - 1)


# Just to store some initial variables and calculations
class RecEngine:
    def __init__(self):
        # Movie Vectors we created in RecommendationEngine.ipynb
        self.db = pd.read_csv(filepath_or_buffer="./MovieVectors.csv",
                              sep=",",
                              names=['MovieID', 'MovieName', 'Vector'], header=0, index_col='MovieID')
        # Convert string to list
        self.db['Vector'] = self.db['Vector'].apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])

        # Start app with asyncio support
        self.app = QApplication(sys.argv)
        loop = QEventLoop(self.app)
        asyncio.set_event_loop(loop)

        self.window = MainWindow(options=self.db['MovieName'].tolist(), db=self.db)
        self.window.show()

        with loop:
            loop.run_forever()

if __name__ == "__main__":
    rec = RecEngine()
