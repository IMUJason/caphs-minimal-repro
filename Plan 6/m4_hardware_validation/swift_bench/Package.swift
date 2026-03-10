// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "M4HeteroBench",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .executable(name: "M4HeteroBench", targets: ["M4HeteroBench"]),
    ],
    targets: [
        .executableTarget(
            name: "M4HeteroBench",
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Accelerate"),
            ]
        ),
    ]
)
