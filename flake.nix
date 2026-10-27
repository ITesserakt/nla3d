{
  description = "2D/3D finite element programming framework";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixpkgs-unstable/nixexprs.tar.zst";
    # This is old nixpkgs with correct deps versions
    site-deps-nixpkgs.url = "github:NixOS/nixpkgs/14aef06d9b3ad1d07626bdbb16083b83f92dc6c1";
  };

  outputs = inputs: {
    packages = builtins.mapAttrs (
      system: pkgs:
      let
        pname = "nla3d";
        version = "0.0.0";
        src = ./.;
        python = pkgs.python3;
        site-deps-pkgs = import inputs.site-deps-nixpkgs { 
          inherit system;
        };
      in
      {
        nla3d = pkgs.stdenv.mkDerivation {
          inherit pname version src;
          nativeBuildInputs = [ 
            pkgs.cmake 
            pkgs.pkg-config 
          ];
          buildInputs = with site-deps-pkgs; [ 
            easyloggingpp 
            eigen 
          ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
            "-DUSE_PKG_CONFIG=ON"
          ];
        };

        nla3d_py = python.pkgs.buildPythonPackage {
          pname = "nla3d";
          inherit version src;

          nativeBuildInputs = [
            pkgs.cmake
            pkgs.swig
            pkgs.pkg-config
          ];
          buildInputs = with site-deps-pkgs; [ 
            python 
            easyloggingpp
            eigen
          ];
          propagatedBuildInputs = [ python ];

          pyproject = false;

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
            "-DNLA3D_PYTHON=ON"
            "-DUSE_PKG_CONFIG=ON"
          ];

          buildPhase = ''
            runHook preBuild
            cmake --build . --target nla3d_py
            runHook postBuild
          '';

          installPhase =
            let
              site-packages = python.sitePackages;
              extension_suffix = builtins.readFile (
                pkgs.runCommandNoCC "python-extension-suffix"
                  {
                    nativeBuildInputs = [ python ];
                  }
                  ''
                    python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"), end="")' > $out
                  ''
              );
            in
            ''
              runHook preInstall

              mkdir -p $out/${site-packages}

              cp python/_nla3d${extension_suffix} python/nla3d.py $out/${site-packages}/

              runHook postInstall
            '';
        };

        nla3d-with-mkl = pkgs.stdenv.mkDerivation {
          inherit pname version src;
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.pkg-config
          ];
          buildInputs = with site-deps-pkgs; [ 
            mkl
            easyloggingpp
            eigen
          ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
            "-DNLA3D_USE_MKL=ON"
            "-DUSE_PKG_CONFIG=ON"
          ];
        };

        default = inputs.self.packages.${system}.nla3d;
      }
    ) inputs.nixpkgs.legacyPackages;

    devShells = builtins.mapAttrs (system: pkgs: {
      default = pkgs.mkShellNoCC {
        packages =
          with pkgs;
          [
            llvmPackages.clang
            llvmPackages.openmp
            gdb
            cmake
            swig
            pkg-config
            python3
          ]
          ++ pkgs.lib.optional (pkgs.stdenv.hostPlatform.is64bit && pkgs.stdenv.hostPlatform.isLinux) pkgs.mkl
          ++ (with import inputs.site-deps-nixpkgs { inherit system; }; [
            easyloggingpp
            eigen
          ]);
      };
    }) inputs.nixpkgs.legacyPackages;
  };
}
