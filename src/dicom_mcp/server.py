import asyncio
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl, BaseModel
import mcp.server.stdio

# Import our DICOM processing tools
from dicom_mcp.dicom_tools import (
    scan_directory, 
    load_dicom_image, 
    load_dicom_seg,
    extract_dicom_metadata, 
    crop_image,
    crop_itk_image,
    DicomSeries
)

# Storage for DICOM series and other data
notes: dict[str, str] = {}  # Keep the notes functionality for backward compatibility
dicom_data: Dict[str, DicomSeries] = {}  # Store DICOM series info
dicom_cache: Dict[str, Any] = {}  # Cache for loaded images and metadata

server = Server("DICOM-MCP")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available resources including notes and DICOM series.
    Resources are exposed with appropriate URI schemes (note:// for notes, dicom:// for DICOM).
    """
    resources = [
        types.Resource(
            uri=AnyUrl(f"note://internal/{name}"),
            name=f"Note: {name}",
            description=f"A simple note named {name}",
            mimeType="text/plain",
        )
        for name in notes
    ]
    
    # Add DICOM series as resources
    for series_uid, series in dicom_data.items():
        modality = series.modality or "Unknown"
        description = series.description or f"DICOM Series {series_uid}"
        resources.append(
            types.Resource(
                uri=AnyUrl(f"dicom://series/{series_uid}"),
                name=f"{modality}: {description}",
                description=f"DICOM Series with {series.file_count} files. Patient ID: {series.patient_id or 'Unknown'}",
                mimeType="application/dicom",
            )
        )
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> Union[str, types.TextContent, types.ImageContent]:
    """
    Read a specific resource's content by its URI.
    Supports both note:// and dicom:// URI schemes.
    """
    scheme = uri.scheme
    path = uri.path.lstrip("/") if uri.path else ""
    
    if scheme == "note":
        if path in notes:
            return notes[path]
        raise ValueError(f"Note not found: {path}")
        
    elif scheme == "dicom":
        # Extract series UID from the path
        parts = path.split("/")
        if len(parts) < 2 or parts[0] != "series":
            raise ValueError(f"Invalid DICOM URI format: {uri}")
            
        series_uid = parts[1]
        if series_uid not in dicom_data:
            raise ValueError(f"DICOM series not found: {series_uid}")
        
        # For now, return metadata as JSON string
        series = dicom_data[series_uid]
        return types.TextContent(
            type="text",
            text=json.dumps(series.dict(), indent=2)
        )
        
    else:
        raise ValueError(f"Unsupported URI scheme: {scheme}")

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return [
        types.Prompt(
            name="summarize-notes",
            description="Creates a summary of all notes",
            arguments=[
                types.PromptArgument(
                    name="style",
                    description="Style of the summary (brief/detailed)",
                    required=False,
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    The prompt includes all current notes and can be customized via arguments.
    """
    if name != "summarize-notes":
        raise ValueError(f"Unknown prompt: {name}")

    style = (arguments or {}).get("style", "brief")
    detail_prompt = " Give extensive details." if style == "detailed" else ""

    return types.GetPromptResult(
        description="Summarize the current notes",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Here are the current notes to summarize:{detail_prompt}\n\n"
                    + "\n".join(
                        f"- {name}: {content}"
                        for name, content in notes.items()
                    ),
                ),
            )
        ],
    )

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools including DICOM-specific tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="add-note",
            description="Add a new note",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
            },
        ),
        types.Tool(
            name="scan-dicom-directory",
            description="Scan a directory for DICOM files and organize them into series",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Path to directory containing DICOM files"},
                },
                "required": ["directory"],
            },
        ),
        types.Tool(
            name="extract-dicom-metadata",
            description="Extract detailed metadata from a DICOM file",
            inputSchema={
                "type": "object",
                "properties": {
                    "dicom_file": {"type": "string", "description": "Path to a DICOM file"},
                },
                "required": ["dicom_file"],
            },
        ),
        types.Tool(
            name="load-dicom-series",
            description="Load a DICOM series into memory for processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "series_uid": {"type": "string", "description": "Series UID of the DICOM series to load"},
                },
                "required": ["series_uid"],
            },
        ),
        types.Tool(
            name="load-dicom-seg",
            description="Load a DICOM SEG file and associate it with a reference image",
            inputSchema={
                "type": "object",
                "properties": {
                    "seg_file": {"type": "string", "description": "Path to a DICOM SEG file"},
                    "reference_series_uid": {"type": "string", "description": "Series UID of the reference image"},
                },
                "required": ["seg_file"],
            },
        ),
        types.Tool(
            name="crop-dicom-image",
            description="Crop a loaded DICOM image by removing boundary percentage",
            inputSchema={
                "type": "object",
                "properties": {
                    "series_uid": {"type": "string", "description": "Series UID of the loaded DICOM image"},
                    "boundary_percentage": {"type": "number", "description": "Percentage of image to crop from each boundary (0.0-0.5)"},
                },
                "required": ["series_uid"],
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests for both basic note tools and DICOM-specific tools.
    Tools can modify server state and notify clients of changes.
    """
    if not arguments:
        raise ValueError("Missing arguments")
        
    # Handle the original add-note tool
    if name == "add-note":
        note_name = arguments.get("name")
        content = arguments.get("content")

        if not note_name or not content:
            raise ValueError("Missing name or content")

        # Update server state
        notes[note_name] = content

        # Notify clients that resources have changed
        await server.request_context.session.send_resource_list_changed()

        return [
            types.TextContent(
                type="text",
                text=f"Added note '{note_name}' with content: {content}",
            )
        ]
    
    # DICOM Tools
    elif name == "scan-dicom-directory":
        directory = arguments.get("directory")
        if not directory or not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}")
            
        # Scan the directory for DICOM files
        series_list = scan_directory(directory)
        
        # Update state with found series
        for series in series_list:
            dicom_data[series.series_uid] = series
            
        # Notify clients that resources have changed
        await server.request_context.session.send_resource_list_changed()
        
        return [
            types.TextContent(
                type="text",
                text=f"Found {len(series_list)} DICOM series in {directory}:\n" + 
                     "\n".join([f"- {s.modality or 'Unknown'}: {s.description or s.series_uid} ({s.file_count} files)" for s in series_list])
            )
        ]
    
    elif name == "extract-dicom-metadata":
        dicom_file = arguments.get("dicom_file")
        if not dicom_file or not os.path.isfile(dicom_file):
            raise ValueError(f"Invalid DICOM file: {dicom_file}")
            
        # Extract metadata
        metadata = extract_dicom_metadata(dicom_file)
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps(metadata, indent=2, default=str)
            )
        ]
    
    elif name == "load-dicom-series":
        series_uid = arguments.get("series_uid")
        if not series_uid or series_uid not in dicom_data:
            raise ValueError(f"Invalid or unknown series UID: {series_uid}")
            
        series = dicom_data[series_uid]
        
        # Load the DICOM series
        image_array, metadata = load_dicom_image(series.path)
        
        # Cache the loaded data
        dicom_cache[series_uid] = {
            "image": image_array,
            "metadata": metadata
        }
        
        # Return summary information
        shape = image_array.shape
        intensity_range = (float(image_array.min()), float(image_array.max()))
        
        return [
            types.TextContent(
                type="text",
                text=f"Loaded DICOM series {series_uid}\n" +
                     f"Shape: {shape}\n" +
                     f"Intensity range: {intensity_range}\n" +
                     f"Metadata: {json.dumps(metadata, indent=2, default=str)}"
            )
        ]
    
    elif name == "load-dicom-seg":
        seg_file = arguments.get("seg_file")
        reference_series_uid = arguments.get("reference_series_uid")
        
        if not seg_file or not os.path.isfile(seg_file):
            raise ValueError(f"Invalid DICOM SEG file: {seg_file}")
            
        # Get reference image if provided
        reference_image = None
        if reference_series_uid:
            if reference_series_uid not in dicom_cache:
                raise ValueError(f"Reference series {reference_series_uid} not loaded. Use load-dicom-series first.")
            reference_image = dicom_cache[reference_series_uid]["image"]
            
        # Load the DICOM SEG
        seg_array, seg_metadata = load_dicom_seg(seg_file, reference_image)
        
        # Generate a unique ID for this segmentation
        seg_uid = f"seg_{Path(seg_file).stem}"
        
        # Cache the loaded data
        dicom_cache[seg_uid] = {
            "image": seg_array,
            "metadata": seg_metadata,
            "reference": reference_series_uid
        }
        
        # Return summary information
        shape = seg_array.shape
        segment_count = len(seg_metadata.get("segment_info", []))
        
        return [
            types.TextContent(
                type="text",
                text=f"Loaded DICOM SEG {seg_uid}\n" +
                     f"Shape: {shape}\n" +
                     f"Segments: {segment_count}\n" +
                     f"Metadata: {json.dumps(seg_metadata, indent=2, default=str)}"
            )
        ]
    
    elif name == "crop-dicom-image":
        series_uid = arguments.get("series_uid")
        boundary_percentage = float(arguments.get("boundary_percentage", 0.2))
        
        if not series_uid or series_uid not in dicom_cache:
            raise ValueError(f"Invalid or not loaded series UID: {series_uid}. Use load-dicom-series first.")
            
        if boundary_percentage <= 0 or boundary_percentage >= 0.5:
            raise ValueError("Boundary percentage must be between 0 and 0.5")
            
        # Get the cached image
        cached_data = dicom_cache[series_uid]
        original_image = cached_data["image"]
        
        # Crop the image
        cropped_image = crop_image(original_image, boundary_percentage)
        
        # Cache the cropped image with a new ID
        cropped_uid = f"{series_uid}_cropped_{int(boundary_percentage*100)}"
        dicom_cache[cropped_uid] = {
            "image": cropped_image,
            "metadata": cached_data["metadata"],
            "original": series_uid,
            "crop_percentage": boundary_percentage
        }
        
        # Return summary information
        original_shape = original_image.shape
        cropped_shape = cropped_image.shape
        
        return [
            types.TextContent(
                type="text",
                text=f"Cropped DICOM image {series_uid} to {cropped_uid}\n" +
                     f"Original shape: {original_shape}\n" +
                     f"Cropped shape: {cropped_shape}\n" +
                     f"Boundary percentage: {boundary_percentage}"
            )
        ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="DICOM-MCP",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )