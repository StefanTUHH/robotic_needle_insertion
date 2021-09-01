cmake_minimum_required(VERSION 3.5)

project(NeedleInsertion)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/NeedleInsertion")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Stefan Gerlach (TUHH), Anonymous")
set(EXTENSION_DESCRIPTION "This is an extension for automated needle insertion")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/NeedleInsertion.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/NeedleInsertion/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BestPathVisualization)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
