import subprocess
from pathlib import Path
from typing import List
from RPA.Desktop import utils
from RPA.Desktop.keywords import LibraryContext, keyword


class Application:
    """Container for launched application."""

    def __init__(self, name: str, args: List[str], shell: bool = False):
        self._name = name
        self._args = args
        self._shell = shell
        self._proc = None

    def __str__(self):
        return 'Application("{name}", pid:{pid})'.format(name=self._name, pid=self.pid)

    @property
    def is_running(self):
        if not self._proc:
            return False

        return self._proc.poll() is None

    @property
    def pid(self):
        return self._proc.pid if self._proc else None

    def start(self):
        if self._proc:
            raise RuntimeError("Application already started")

        self._proc = subprocess.Popen(self._args, shell=self._shell)

    def stop(self):
        if self._proc:
            self._proc.terminate()

    def wait(self, timeout=30):
        if not self._proc:
            raise RuntimeError("Application not started")

        self._proc.communicate(timeout=int(timeout))


class ApplicationKeywords(LibraryContext):
    """Keywords for starting and stopping applications."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self._apps = []

    def _create_app(self, name: str, args: List[str], shell: bool = False):
        app = Application(name, args, shell)
        app.start()

        self._apps.append(app)
        return app

    @keyword
    def open_application(self, name_or_path: str, *args) -> Application:
        """Start a given application by name (if in PATH),
        or by path to executable.

        Example:

        .. code-block:: robotframework

            Open application    notepad.exe
            Open application    c:\\path\\to\\program.exe    --example-argument

        :param name_or_path: Name or path of application
        :param *args:        List of command line arguments for application
        :returns:            Application instance
        """
        name = Path(name_or_path).name
        return self._create_app(name, [name_or_path] + list(args))

    @keyword
    def open_file(self, path: str) -> Application:
        """Open a file with the default application.

        Example:

        .. code-block:: robotframework

            Open file    orders.xlsx

        :param path: Path to file
        """
        name = Path(path).name

        if utils.is_windows():
            return self._create_app(name, ["start", "/WAIT"], shell=True)
        elif utils.is_macos():
            return self._create_app(name, ["open", "-W", path])
        else:
            # TODO: xdg-open quits immediately after child process has started,
            # figure out default application some other way and launch directly.
            return self._create_app(name, ["xdg-open", path])

    @keyword
    def close_application(self, app: Application) -> None:
        """Close given application. Needs to be started
        with this library.

        Example:

        .. code-block:: robotframework

            ${word}=    Open file    template.docx
            # Do something with Word
            Close application    ${word}

        :param app: App instance
        """
        if app.is_running:
            app.stop()

    @keyword
    def close_all_applications(self) -> None:
        """Close all opened applications.

        Example:

        .. code-block:: robotframework

            Open file    order1.docx
            Open file    order2.docx
            Open file    order3.docx
            # Do something with Word
            Close all applications
        """
        for app in self._apps:
            if app.is_running:
                app.stop()
