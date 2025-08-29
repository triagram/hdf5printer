import argparse
import h5py
import numpy as np
from typing import Any, TextIO, Union, Optional


class HDF5Explorer:

    def __init__(self, max_display_items: int = 10, max_string_length: int = 100):

        self.max_display_items = max_display_items
        self.max_string_length = max_string_length

    def format_dataset_content(self, data: Any) -> str:
 
        if isinstance(data, np.ndarray):
            return self._format_numpy_array(data)
        elif isinstance(data, bytes):
            return self._format_bytes_data(data)
        elif isinstance(data, str):
            return self._format_string_data(data)
        else:
            return str(data)

    def _format_numpy_array(self, array: np.ndarray) -> str:
        """Format numpy arrays with size-aware display."""
        if array.size == 0:
            return "[] (empty array)"

        if array.size <= self.max_display_items:
            return str(array)

        if array.ndim == 1:
            half_items = self.max_display_items // 2
            head = array[:half_items]
            tail = array[-half_items:]
            head_str = ", ".join(map(str, head))
            tail_str = ", ".join(map(str, tail))
            return (
                f"[{head_str} ... {tail_str}] "
                f"(shape={array.shape}, first/last {self.max_display_items} items)"
            )
        else:
            return (
                f"<array of shape {array.shape} dtype={array.dtype}> "
                f"(too large to display)"
            )

    def _format_bytes_data(self, data: bytes) -> str:
        """Format bytes data with length-aware display."""
        if len(data) < self.max_string_length:
            return data.decode("utf-8", "replace")
        else:
            return f"<bytes of length {len(data)}>"

    def _format_string_data(self, data: str) -> str:
        """Format string data with truncation for long strings."""
        if len(data) <= self.max_string_length:
            return data
        else:
            return data[: self.max_string_length] + "... [truncated]"

    def print_to_console_and_file(
        self, message: str, file_handle: Optional[TextIO] = None
    ) -> None:
        """
        Print a message to console and optionally to file.

        Args:
            message: The message to print
            file_handle: File handle for output (optional)
        """
        print(message)
        if file_handle is not None:
            print(message, file=file_handle)

    def traverse_hdf5_group(
        self,
        group: h5py.Group,
        file_handle: Optional[TextIO] = None,
        indent_level: int = 0,
    ) -> None:
        """
        Recursively traverse HDF5 groups and datasets.

        Args:
            group: The HDF5 group to traverse
            file_handle: File handle for output (optional)
            indent_level: Current indentation level
        """
        indent = "  " * indent_level
        self.print_to_console_and_file(f"{indent}Group: {group.name}", file_handle)

        # Display group attributes
        self._display_attributes(group.attrs, indent, file_handle)

        # Traverse group members
        for name, item in group.items():
            self._process_group_item(item, name, indent, file_handle, indent_level)

    def _display_attributes(
        self, attrs: h5py.AttributeManager, indent: str, file_handle: Optional[TextIO]
    ) -> None:
        """Display attributes of a group or dataset."""
        for attr_name, attr_value in attrs.items():
            formatted_value = self.format_dataset_content(attr_value)
            self.print_to_console_and_file(
                f"{indent}  ├─ Attribute: {attr_name} = {formatted_value}", file_handle
            )

    def _process_group_item(
        self,
        item: Union[h5py.Group, h5py.Dataset],
        name: str,
        indent: str,
        file_handle: Optional[TextIO],
        indent_level: int,
    ) -> None:
        """Process a single item within an HDF5 group."""
        sub_indent = indent + "  ├─ "

        if isinstance(item, h5py.Group):
            self._process_subgroup(item, name, sub_indent, file_handle, indent_level)
        elif isinstance(item, h5py.Dataset):
            self._process_dataset(item, name, sub_indent, indent, file_handle)

    def _process_subgroup(
        self,
        item: h5py.Group,
        name: str,
        sub_indent: str,
        file_handle: Optional[TextIO],
        indent_level: int,
    ) -> None:
        """Process an HDF5 subgroup."""
        self.print_to_console_and_file(f"{sub_indent}Subgroup: {name}", file_handle)
        self.traverse_hdf5_group(item, file_handle, indent_level + 2)

    def _process_dataset(
        self,
        item: h5py.Dataset,
        name: str,
        sub_indent: str,
        indent: str,
        file_handle: Optional[TextIO],
    ) -> None:
        self.print_to_console_and_file(f"{sub_indent}Dataset: {name}", file_handle)

        # Display dataset metadata
        metadata_indent = indent + "  │    ├─ "
        self.print_to_console_and_file(
            f"{metadata_indent}Path: {item.name}", file_handle
        )
        self.print_to_console_and_file(
            f"{metadata_indent}Shape: {item.shape}", file_handle
        )
        self.print_to_console_and_file(
            f"{metadata_indent}Dtype: {item.dtype}", file_handle
        )
        self.print_to_console_and_file(
            f"{metadata_indent}Size: {item.size} elements", file_handle
        )

        # Display dataset attributes
        self._display_attributes(item.attrs, indent + "  │  ", file_handle)

        # Display dataset content safely
        self._display_dataset_content(item, indent, file_handle)

    def _display_dataset_content(
        self, dataset: h5py.Dataset, indent: str, file_handle: Optional[TextIO]
    ) -> None:
        """Safely display dataset content."""
        try:
            if dataset.size <= self.max_display_items:
                data = dataset[()]
                formatted_data = self.format_dataset_content(data)
                self.print_to_console_and_file(
                    f"{indent}  │    └─ Data: {formatted_data}", file_handle
                )
            else:
                self.print_to_console_and_file(
                    f"{indent}  │    └─ Data: <too large ({dataset.size} elements), "
                    f"skipping display>",
                    file_handle,
                )
        except Exception as exc:
            self.print_to_console_and_file(
                f"{indent}  │    └─ Data: <error reading: {exc}>", file_handle
            )

    def explore_file_structure(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        save_to_file: bool = True,
    ) -> None:

        try:
            if save_to_file and output_path:
                with h5py.File(file_path, "r") as h5file, open(
                    output_path, "w"
                ) as out_file:
                    self._explore_with_file(h5file, file_path, out_file, output_path)
            else:
                with h5py.File(file_path, "r") as h5file:
                    self._explore_console_only(h5file, file_path)

        except FileNotFoundError:
            print(f"Error: HDF5 file '{file_path}' not found.")
        except OSError as exc:
            print(f"Error: Cannot read HDF5 file '{file_path}': {exc}")
        except Exception as exc:
            print(f"Unexpected error: {exc}")

    def _explore_with_file(
        self, h5file: h5py.File, file_path: str, out_file: TextIO, output_path: str
    ) -> None:

        self.print_to_console_and_file("=" * 60, out_file)

        # Traverse the file structure
        self.traverse_hdf5_group(h5file, out_file, indent_level=0)

        print(f"Results saved to: {output_path}")

    def _explore_console_only(self, h5file: h5py.File, file_path: str) -> None:
        """Explore HDF5 file with console output only."""
        # Print header
        self.print_to_console_and_file(f"HDF5 File Structure: {file_path}")
        self.print_to_console_and_file("=" * 60)

        # Traverse the file structure
        self.traverse_hdf5_group(h5file, None, indent_level=0)

        # Print footer
        self.print_to_console_and_file("\nStructure exploration completed.")


def create_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Explore HDF5 file structure with safe content display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.h5
  %(prog)s data.h5 -o structure_report.txt -m 10
        """,
    )

    parser.add_argument("h5_file", type=str, help="Path to the HDF5 file to explore")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="h5_structure.txt",
        help="Output file full path (default: h5_structure.txt)",
    )
    parser.add_argument(
        "-m",
        "--max-items",
        type=int,
        default=10,
        help="Maximum items to display from datasets (default: 10)",
    )
    parser.add_argument(
        "-l",
        "--max-string-length",
        type=int,
        default=100,
        help="Maximum string length before truncation (default: 100)",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save output to file instead of displaying in console",
    )

    return parser


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    print(f"current HDF5 directory: ./{args.h5_file}")

    explorer = HDF5Explorer(
        max_display_items=args.max_items, max_string_length=args.max_string_length
    )

    save_to_file = args.save

    if save_to_file:
        explorer.explore_file_structure(args.h5_file, args.output, save_to_file=True)
    else:
        explorer.explore_file_structure(args.h5_file, save_to_file=False)


if __name__ == "__main__":
    main()
