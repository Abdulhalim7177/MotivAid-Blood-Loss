# Requirements Document: Dataset Quality Improvement

## Introduction

The MotivAid Blood Loss AI system is a medical application that estimates blood loss volume from smartphone photographs using a two-stage deep learning pipeline (segmentation + regression). The system currently faces critical data quality issues that prevent successful training and deployment. This feature addresses dataset labeling organization, data quality problems, and surface type standardization to enable reliable model training and evaluation.

The current dataset contains diverse surface types (bowl, container, pad, pampers, drape, floor, cloth, bedsheet, towel, and compound surfaces) that need to be properly categorized and validated. The system needs to support all these surface types while maintaining compatibility with the training pipeline. Additionally, the dataset contains potential outliers (3 mL to 11,000 mL range) that may corrupt model learning.

## Glossary

- **Dataset**: Collection of 339 training images in dataset/synthetic_train/ with associated volume and surface type labels
- **Surface_Type**: Category of material containing blood (bowl, container, pad, pampers, drape, floor, cloth, bedsheet, towel, gauze, sheet, other, and compound types)
- **Training_Script**: Python scripts (train_seg.py, train_reg.py) that train the segmentation and regression models
- **Label_File**: JSON file (synthetic_labels.json) mapping image filenames to volume_ml and surface_type
- **Data_Quality_Report**: Analysis document identifying outliers, duplicates, and labeling issues
- **Surface_Standardization_Script**: Automated tool that validates and standardizes surface type labels
- **Outlier**: Data point with volume outside expected physiological range or statistical distribution
- **Duplicate_Image**: Multiple images with identical or near-identical pixel content
- **Image_Compression**: Image quality degradation from export/transfer processes
- **Segmentation_Model**: First-stage model that identifies blood-stained pixels in images
- **Regression_Model**: Second-stage model that estimates blood volume in milliliters
- **Clinical_Surface**: Medical materials used in healthcare settings (gauze, hospital sheets, surgical drapes, sanitary pads)
- **Compound_Surface**: Images containing multiple surface types (e.g., floor-and-cloth, pad-and-container)

## Requirements

### Requirement 1: Surface Type Standardization and Validation

**User Story:** As a data scientist, I want all dataset surface types validated and standardized, so that training scripts work with the actual surface categories in the dataset.

#### Acceptance Criteria

1. THE Surface_Standardization_Script SHALL parse all filenames in dataset/synthetic_train/ and identify current surface type labels
2. THE Surface_Standardization_Script SHALL support all surface types found in the dataset: bowl, container, pad, pampers, drape, floor, cloth, bedsheet, towel, gauze, sheet, other
3. THE Surface_Standardization_Script SHALL support compound surface types: floor-and-cloth, cloth-and-floor, pad-and-container, pad-and-floor
4. THE Surface_Standardization_Script SHALL generate updated synthetic_labels.json with validated surface types
5. THE Surface_Standardization_Script SHALL preserve original volume_ml values during processing
6. WHEN standardization is complete, THE Surface_Standardization_Script SHALL generate a report showing surface type distribution
7. THE Surface_Standardization_Script SHALL create a backup of original labels before modification
8. FOR ALL images in the dataset, parsing the standardized labels then validating surface types SHALL confirm all categories are recognized (round-trip validation property)

### Requirement 2: Data Quality Analysis

**User Story:** As a researcher, I want automated data quality checks, so that outliers, duplicates, and labeling errors are identified before training.

#### Acceptance Criteria

1. THE Quality_Analyzer SHALL scan all images in dataset/synthetic_train/
2. THE Quality_Analyzer SHALL identify volume outliers using statistical methods (values beyond 3 standard deviations from mean)
3. THE Quality_Analyzer SHALL identify physiological outliers (volumes below 3mL or above 2000mL for typical postpartum scenarios)
4. THE Quality_Analyzer SHALL detect duplicate images using perceptual hashing (images with >95% similarity)
5. THE Quality_Analyzer SHALL detect image compression artifacts by analyzing JPEG quality metrics
6. THE Quality_Analyzer SHALL identify images with missing or corrupted label entries
7. THE Quality_Analyzer SHALL calculate volume distribution statistics (min, max, mean, median, quartiles) per surface type
8. THE Quality_Analyzer SHALL generate a Data_Quality_Report in markdown format with findings and recommendations
9. WHEN extreme outliers are detected (>2000mL or <3mL), THE Quality_Analyzer SHALL flag them for manual review
10. THE Quality_Analyzer SHALL provide actionable recommendations for each identified issue

### Requirement 3: Surface Type Category Management

**User Story:** As a data scientist, I want clear, medically-informed remapping rules, so that non-standard surface types are converted to appropriate clinical categories.

#### Acceptance Criteria

1. THE Remapping_Rules SHALL map "bowl" to "other" (non-clinical collection vessel)
2. THE Remapping_Rules SHALL map "container" to "other" (non-clinical collection vessel)
3. THE Remapping_Rules SHALL map "pampers" to "pad" (absorbent material similar to sanitary pad)
4. THE Remapping_Rules SHALL map "floor" to "other" (non-absorbent surface)
5. THE Remapping_Rules SHALL map "cloth" to "sheet" (fabric material similar to hospital sheet)
6. THE Remapping_Rules SHALL map compound labels "floor-and-cloth" to "other" (mixed non-clinical surfaces)
7. THE Remapping_Rules SHALL map compound labels "cloth-and-floor" to "other" (mixed non-clinical surfaces)
8. THE Remapping_Rules SHALL map compound labels "pad-and-container" to "pad" (primary absorbent surface)
9. THE Remapping_Rules SHALL map compound labels "pad-and-floor" to "pad" (primary absorbent surface)
10. THE Remapping_Rules SHALL be documented in a configuration file (remapping_rules.json) for easy modification
11. THE Remapping_Rules SHALL include rationale comments explaining each mapping decision

### Requirement 4: Missing Surface Type Documentation

**User Story:** As a medical professional, I want documentation on missing clinical surface types, so that I understand data collection gaps and can plan additional data acquisition.

#### Acceptance Criteria

1. THE Documentation_Generator SHALL identify surface types with zero samples in the dataset
2. THE Documentation_Generator SHALL identify surface types with insufficient samples (<10 images)
3. THE Documentation_Generator SHALL generate a missing_surfaces_report.md documenting gaps
4. THE Missing_Surfaces_Report SHALL include recommended sample counts for each missing surface type
5. THE Missing_Surfaces_Report SHALL include photography guidelines for gauze (4x4 surgical sponges)
6. THE Missing_Surfaces_Report SHALL include photography guidelines for hospital sheets (white cotton)
7. THE Missing_Surfaces_Report SHALL reference the blood analog recipe from the implementation guide
8. THE Missing_Surfaces_Report SHALL specify volume ranges to photograph for each missing surface type
9. THE Missing_Surfaces_Report SHALL specify lighting and distance requirements (20cm, 40cm; daylight, LED, fluorescent)

### Requirement 5: Backward Compatibility

**User Story:** As a developer, I want data quality improvements to maintain compatibility with existing training scripts, so that no code changes are required in the training pipeline.

#### Acceptance Criteria

1. THE Updated_Label_Files SHALL maintain the same JSON structure as original synthetic_labels.json
2. THE Updated_Label_Files SHALL use identical key names ("volume_ml", "surface_type")
3. THE Updated_Label_Files SHALL preserve the nested structure with split names as top-level keys
4. WHEN Training_Script loads updated labels, THE Training_Script SHALL execute without modification
5. THE Surface_Type values SHALL match the SURFACE_MAP dictionary in scripts/dataset.py exactly
6. FOR ALL training scripts, running with updated data SHALL produce no import or key errors

### Requirement 6: Non-Destructive Operations

**User Story:** As a data scientist, I want all data modifications to preserve original images, so that I can revert changes if needed.

#### Acceptance Criteria

1. THE Surface_Standardization_Script SHALL create a backup directory (dataset/backup_labels/) before modifying labels
2. THE Surface_Standardization_Script SHALL copy original synthetic_labels.json to backup directory
3. THE Quality_Analyzer SHALL operate in read-only mode (no file modifications)
4. WHEN any script encounters an error, THE Script SHALL restore original state from backup
5. THE Scripts SHALL log all file operations to an audit trail (data_operations.log)
6. THE Scripts SHALL provide a rollback command to restore original dataset structure

### Requirement 7: Automated Validation

**User Story:** As a developer, I want automated validation after data processing, so that I can verify the dataset is ready for training.

#### Acceptance Criteria

1. THE Validation_Script SHALL verify all images in synthetic_train/ have corresponding labels
2. THE Validation_Script SHALL verify all surface_type values are recognized by the system
3. THE Validation_Script SHALL verify all volume_ml values are positive numbers
4. THE Validation_Script SHALL verify each surface type has at least 1 sample in the dataset
5. THE Validation_Script SHALL attempt to load data using scripts/dataset.py to confirm compatibility
6. WHEN validation passes, THE Validation_Script SHALL output "DATASET READY FOR TRAINING"
7. WHEN validation fails, THE Validation_Script SHALL output specific error messages with remediation steps
8. THE Validation_Script SHALL generate a validation_report.json with all check results

### Requirement 8: Windows Compatibility

**User Story:** As a developer working on Windows, I want all scripts to run correctly in Windows bash shell, so that I can process data on my development machine.

#### Acceptance Criteria

1. THE Scripts SHALL use os.path.join() for all file path construction (not string concatenation)
2. THE Scripts SHALL use pathlib.Path for cross-platform path operations where appropriate
3. THE Scripts SHALL avoid Unix-specific shell commands (grep, find, awk)
4. THE Scripts SHALL handle Windows path separators (backslash) correctly
5. WHEN run on Windows with bash shell, THE Scripts SHALL execute without path-related errors
6. THE Scripts SHALL use Python 3.10+ compatible syntax and libraries
7. THE Scripts SHALL include shebang lines for bash execution (#!/usr/bin/env python)

### Requirement 9: Execution Workflow

**User Story:** As a data scientist, I want a clear step-by-step workflow, so that I can process the dataset correctly without missing steps.

#### Acceptance Criteria

1. THE Workflow_Documentation SHALL specify the exact order of script execution
2. THE Workflow_Documentation SHALL include checkpoint verification steps after each script
3. THE Workflow_Documentation SHALL specify expected outputs for each script
4. THE Workflow_Documentation SHALL include troubleshooting guidance for common errors
5. THE Workflow_Documentation SHALL be provided in WORKFLOW.md in the project root
6. THE Workflow_Documentation SHALL include estimated execution time for each step
7. THE Workflow_Documentation SHALL specify when to run scripts locally vs. in Google Colab

### Requirement 10: Data Quality Metrics

**User Story:** As a researcher, I want quantitative data quality metrics, so that I can assess dataset readiness objectively.

#### Acceptance Criteria

1. THE Quality_Analyzer SHALL calculate percentage of outliers in the dataset
2. THE Quality_Analyzer SHALL calculate percentage of duplicate images
3. THE Quality_Analyzer SHALL calculate surface type distribution (percentage per category)
4. THE Quality_Analyzer SHALL calculate volume range coverage (percentage in low/medium/high ranges)
5. THE Quality_Analyzer SHALL calculate average image resolution
6. THE Quality_Analyzer SHALL calculate average JPEG quality score
7. WHEN outlier percentage exceeds 5%, THE Quality_Analyzer SHALL recommend manual review
8. WHEN duplicate percentage exceeds 10%, THE Quality_Analyzer SHALL recommend deduplication
9. WHEN any surface type has less than 5% representation, THE Quality_Analyzer SHALL flag imbalance
10. THE Quality_Analyzer SHALL output metrics in both human-readable (markdown) and machine-readable (JSON) formats

### Requirement 11: Surface Type Configuration

**User Story:** As a data scientist, I want configurable surface type definitions, so that I can add or modify surface categories without changing code.

#### Acceptance Criteria

1. THE Surface_Configuration SHALL be defined in a surface_types.json configuration file
2. THE Configuration_File SHALL list all supported surface types with their properties
3. THE Configuration_File SHALL include a "category" field grouping similar surfaces (clinical, non-clinical, compound)
4. THE Configuration_File SHALL include a "description" field explaining each surface type
5. WHEN surface_types.json is missing, THE Scripts SHALL use built-in default surface types
6. WHEN a new surface type is added to the config, THE Scripts SHALL recognize it automatically
7. THE Scripts SHALL validate that all surface types in the dataset exist in the configuration

### Requirement 12: Training Pipeline Integration

**User Story:** As a model trainer, I want the processed dataset to work seamlessly with existing training scripts, so that I can start training immediately after data preparation.

#### Acceptance Criteria

1. WHEN train_seg.py is executed with processed data, THE Training_Script SHALL load data without errors
2. WHEN train_reg.py is executed with processed data, THE Training_Script SHALL load data without errors
3. THE Training_Scripts SHALL execute successfully with the remapped surface types
4. FOR ALL training epochs, the model SHALL train without surface type key errors

## Success Metrics

The following quantitative metrics define successful implementation:

1. **Zero Training Errors**: train_seg.py and train_reg.py execute without surface type mismatches or key errors
2. **Low Outlier Rate**: Data quality report shows <5% outliers in volume measurements
3. **All Surface Types Supported**: 100% of images have recognized surface_type values
4. **Codebase Updated**: All scripts support the full set of surface types in the dataset
5. **Successful Training Run**: End-to-end training completes without errors and produces model checkpoints
6. **Data Quality Report**: Comprehensive report generated with actionable recommendations

## Technical Constraints

1. **Python Version**: Scripts must run on Python 3.10 or higher
2. **Dependencies**: Must use existing project dependencies (torch, opencv-python, PIL, albumentations, numpy, pandas)
3. **Platform**: Must work on Windows with bash shell environment
4. **File Format**: Labels must remain in JSON format with existing structure
5. **Image Format**: Original JPEG images must not be re-encoded or modified
6. **Memory**: Scripts must process 339 images within 4GB RAM limit
7. **Execution Time**: Data processing should complete within 5 minutes on standard laptop
8. **Backward Compatibility**: Existing training scripts (train_seg.py, train_reg.py, dataset.py) must work without modification

## Out of Scope

The following items are explicitly excluded from this feature:

1. **Model Architecture Changes**: No modifications to segmentation or regression model architectures
2. **New Data Collection**: No acquisition of new photographs (only processing existing 339 images)
3. **Image Augmentation**: No changes to augmentation pipeline in dataset.py
4. **Training Hyperparameters**: No tuning of learning rates, batch sizes, or epoch counts
5. **Real Test Data**: No modifications to dataset/real_test/ or labels.json
6. **Mobile Deployment**: No changes to TFLite conversion or React Native integration
7. **Clinical Validation**: No medical accuracy validation or clinical trial data collection
8. **GUI Tools**: All scripts are command-line only (no graphical interfaces)

## Dependencies

This feature depends on:

1. **Existing Dataset**: 339 images in dataset/synthetic_train/ with embedded metadata in filenames
2. **Python Environment**: Activated virtual environment with required packages installed
3. **Training Scripts**: Existing train_seg.py, train_reg.py, and dataset.py files
4. **File System Access**: Read/write permissions to dataset/ directory
5. **Implementation Guide**: MotivAid_Implementation_Guide.txt for reference specifications

## Acceptance Testing Strategy

Final acceptance testing will verify:

1. **Surface Type Validation**: Run standardization script, verify synthetic_labels.json contains all surface types from dataset
2. **Codebase Update**: Verify SURFACE_MAP in dataset.py includes all surface types with unique indices
3. **Quality Report**: Verify data_quality_report.md is generated with outlier analysis
4. **Training Execution**: Run `python scripts/train_seg.py` and verify it completes epoch 1 without errors
5. **Rollback Test**: Run rollback command and verify original dataset structure is restored
6. **Windows Test**: Execute all scripts on Windows bash and verify no path errors
7. **Documentation Test**: Follow WORKFLOW.md step-by-step and verify all commands execute successfully
