
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * This module provides access to environment variables that are injected _statically_ into your bundle at build time and are limited to _private_ access.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Static environment variables are [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env` at build time and then statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * **_Private_ access:**
 * 
 * - This module cannot be imported into client-side code
 * - This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured)
 * 
 * For example, given the following build time environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { ENVIRONMENT, PUBLIC_BASE_URL } from '$env/static/private';
 * 
 * console.log(ENVIRONMENT); // => "production"
 * console.log(PUBLIC_BASE_URL); // => throws error during build
 * ```
 * 
 * The above values will be the same _even if_ different values for `ENVIRONMENT` or `PUBLIC_BASE_URL` are set at runtime, as they are statically replaced in your code with their build time values.
 */
declare module '$env/static/private' {
	export const SHELL: string;
	export const npm_command: string;
	export const SESSION_MANAGER: string;
	export const WINDOWID: string;
	export const NVM_RC_VERSION: string;
	export const npm_config_userconfig: string;
	export const COLORTERM: string;
	export const OPENCODE_PROCESS_ROLE: string;
	export const XDG_CONFIG_DIRS: string;
	export const npm_config_cache: string;
	export const XDG_SESSION_PATH: string;
	export const XDG_MENU_PREFIX: string;
	export const ICEAUTHORITY: string;
	export const LANGUAGE: string;
	export const NODE: string;
	export const LC_ADDRESS: string;
	export const LC_NAME: string;
	export const SHELL_SESSION_ID: string;
	export const AGENT: string;
	export const MEMORY_PRESSURE_WRITE: string;
	export const COLOR: string;
	export const npm_config_local_prefix: string;
	export const DESKTOP_SESSION: string;
	export const LC_MONETARY: string;
	export const GTK_RC_FILES: string;
	export const npm_config_globalconfig: string;
	export const EDITOR: string;
	export const XDG_SEAT: string;
	export const PWD: string;
	export const XDG_SESSION_DESKTOP: string;
	export const LOGNAME: string;
	export const XDG_SESSION_TYPE: string;
	export const npm_config_init_module: string;
	export const SYSTEMD_EXEC_PID: string;
	export const _: string;
	export const XAUTHORITY: string;
	export const MOTD_SHOWN: string;
	export const GTK2_RC_FILES: string;
	export const OPENCODE_PID: string;
	export const HOME: string;
	export const OPENCODE: string;
	export const LANG: string;
	export const LC_PAPER: string;
	export const _JAVA_AWT_WM_NONREPARENTING: string;
	export const HIPCC_FLAGS: string;
	export const XDG_CURRENT_DESKTOP: string;
	export const KONSOLE_DBUS_SERVICE: string;
	export const npm_package_version: string;
	export const MEMORY_PRESSURE_WATCH: string;
	export const WAYLAND_DISPLAY: string;
	export const KONSOLE_DBUS_SESSION: string;
	export const PROFILEHOME: string;
	export const VIRTUAL_ENV_DISABLE_PROMPT: string;
	export const MANROFFOPT: string;
	export const XDG_SEAT_PATH: string;
	export const OPENCODE_RUN_ID: string;
	export const INVOCATION_ID: string;
	export const HIP_VISIBLE_DEVICES: string;
	export const KONSOLE_VERSION: string;
	export const HSA_ENABLE_SDMA: string;
	export const MANAGERPID: string;
	export const INIT_CWD: string;
	export const MFLAGS: string;
	export const KDE_SESSION_UID: string;
	export const npm_lifecycle_script: string;
	export const NVM_DIR: string;
	export const XKB_DEFAULT_LAYOUT: string;
	export const HIP_PLATFORM: string;
	export const npm_config_npm_version: string;
	export const MAKEFLAGS: string;
	export const XDG_SESSION_CLASS: string;
	export const TERM: string;
	export const LC_IDENTIFICATION: string;
	export const npm_package_name: string;
	export const KONSOLE_DBUS_ACTIVATION_COOKIE: string;
	export const npm_config_prefix: string;
	export const USER: string;
	export const COLORFGBG: string;
	export const QT_WAYLAND_RECONNECT: string;
	export const KDE_SESSION_VERSION: string;
	export const PAM_KWALLET5_LOGIN: string;
	export const MANPAGER: string;
	export const DISPLAY: string;
	export const npm_lifecycle_event: string;
	export const SHLVL: string;
	export const NVM_CD_FLAGS: string;
	export const MAKELEVEL: string;
	export const LC_TELEPHONE: string;
	export const AMD_SERIALIZE_KERNEL: string;
	export const LC_MEASUREMENT: string;
	export const XDG_VTNR: string;
	export const XDG_SESSION_ID: string;
	export const MANAGERPIDFDID: string;
	export const npm_config_user_agent: string;
	export const ROCM_PATH: string;
	export const npm_execpath: string;
	export const LD_LIBRARY_PATH: string;
	export const XDG_RUNTIME_DIR: string;
	export const MKLROOT: string;
	export const DEBUGINFOD_URLS: string;
	export const npm_package_json: string;
	export const LC_TIME: string;
	export const BUN_INSTALL: string;
	export const JOURNAL_STREAM: string;
	export const LLAMA_MTP_PATH: string;
	export const KDE_FULL_SESSION: string;
	export const npm_config_noproxy: string;
	export const PATH: string;
	export const npm_config_node_gyp: string;
	export const HSA_FORCE_FINE_GRAIN_PCIE: string;
	export const HSA_OVERRIDE_GFX_VERSION: string;
	export const DBUS_SESSION_BUS_ADDRESS: string;
	export const npm_config_global_prefix: string;
	export const KDE_APPLICATIONS_AS_SCOPE: string;
	export const MAIL: string;
	export const npm_node_execpath: string;
	export const npm_config_engine_strict: string;
	export const FLATPAK_TTY_PROGRESS: string;
	export const LC_NUMERIC: string;
	export const OLDPWD: string;
	export const KONSOLE_DBUS_WINDOW: string;
	export const HIP_FORCE_WAVEFRONT_SIZE: string;
	export const NODE_ENV: string;
}

/**
 * This module provides access to environment variables that are injected _statically_ into your bundle at build time and are _publicly_ accessible.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Static environment variables are [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env` at build time and then statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * **_Public_ access:**
 * 
 * - This module _can_ be imported into client-side code
 * - **Only** variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`) are included
 * 
 * For example, given the following build time environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { ENVIRONMENT, PUBLIC_BASE_URL } from '$env/static/public';
 * 
 * console.log(ENVIRONMENT); // => throws error during build
 * console.log(PUBLIC_BASE_URL); // => "http://site.com"
 * ```
 * 
 * The above values will be the same _even if_ different values for `ENVIRONMENT` or `PUBLIC_BASE_URL` are set at runtime, as they are statically replaced in your code with their build time values.
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to environment variables set _dynamically_ at runtime and that are limited to _private_ access.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Dynamic environment variables are defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`.
 * 
 * **_Private_ access:**
 * 
 * - This module cannot be imported into client-side code
 * - This module includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured)
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 * 
 * > [!NOTE] To get correct types, environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * >
 * > ```env
 * > MY_FEATURE_FLAG=
 * > ```
 * >
 * > You can override `.env` values from the command line like so:
 * >
 * > ```sh
 * > MY_FEATURE_FLAG="enabled" npm run dev
 * > ```
 * 
 * For example, given the following runtime environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * 
 * console.log(env.ENVIRONMENT); // => "production"
 * console.log(env.PUBLIC_BASE_URL); // => undefined
 * ```
 */
declare module '$env/dynamic/private' {
	export const env: {
		SHELL: string;
		npm_command: string;
		SESSION_MANAGER: string;
		WINDOWID: string;
		NVM_RC_VERSION: string;
		npm_config_userconfig: string;
		COLORTERM: string;
		OPENCODE_PROCESS_ROLE: string;
		XDG_CONFIG_DIRS: string;
		npm_config_cache: string;
		XDG_SESSION_PATH: string;
		XDG_MENU_PREFIX: string;
		ICEAUTHORITY: string;
		LANGUAGE: string;
		NODE: string;
		LC_ADDRESS: string;
		LC_NAME: string;
		SHELL_SESSION_ID: string;
		AGENT: string;
		MEMORY_PRESSURE_WRITE: string;
		COLOR: string;
		npm_config_local_prefix: string;
		DESKTOP_SESSION: string;
		LC_MONETARY: string;
		GTK_RC_FILES: string;
		npm_config_globalconfig: string;
		EDITOR: string;
		XDG_SEAT: string;
		PWD: string;
		XDG_SESSION_DESKTOP: string;
		LOGNAME: string;
		XDG_SESSION_TYPE: string;
		npm_config_init_module: string;
		SYSTEMD_EXEC_PID: string;
		_: string;
		XAUTHORITY: string;
		MOTD_SHOWN: string;
		GTK2_RC_FILES: string;
		OPENCODE_PID: string;
		HOME: string;
		OPENCODE: string;
		LANG: string;
		LC_PAPER: string;
		_JAVA_AWT_WM_NONREPARENTING: string;
		HIPCC_FLAGS: string;
		XDG_CURRENT_DESKTOP: string;
		KONSOLE_DBUS_SERVICE: string;
		npm_package_version: string;
		MEMORY_PRESSURE_WATCH: string;
		WAYLAND_DISPLAY: string;
		KONSOLE_DBUS_SESSION: string;
		PROFILEHOME: string;
		VIRTUAL_ENV_DISABLE_PROMPT: string;
		MANROFFOPT: string;
		XDG_SEAT_PATH: string;
		OPENCODE_RUN_ID: string;
		INVOCATION_ID: string;
		HIP_VISIBLE_DEVICES: string;
		KONSOLE_VERSION: string;
		HSA_ENABLE_SDMA: string;
		MANAGERPID: string;
		INIT_CWD: string;
		MFLAGS: string;
		KDE_SESSION_UID: string;
		npm_lifecycle_script: string;
		NVM_DIR: string;
		XKB_DEFAULT_LAYOUT: string;
		HIP_PLATFORM: string;
		npm_config_npm_version: string;
		MAKEFLAGS: string;
		XDG_SESSION_CLASS: string;
		TERM: string;
		LC_IDENTIFICATION: string;
		npm_package_name: string;
		KONSOLE_DBUS_ACTIVATION_COOKIE: string;
		npm_config_prefix: string;
		USER: string;
		COLORFGBG: string;
		QT_WAYLAND_RECONNECT: string;
		KDE_SESSION_VERSION: string;
		PAM_KWALLET5_LOGIN: string;
		MANPAGER: string;
		DISPLAY: string;
		npm_lifecycle_event: string;
		SHLVL: string;
		NVM_CD_FLAGS: string;
		MAKELEVEL: string;
		LC_TELEPHONE: string;
		AMD_SERIALIZE_KERNEL: string;
		LC_MEASUREMENT: string;
		XDG_VTNR: string;
		XDG_SESSION_ID: string;
		MANAGERPIDFDID: string;
		npm_config_user_agent: string;
		ROCM_PATH: string;
		npm_execpath: string;
		LD_LIBRARY_PATH: string;
		XDG_RUNTIME_DIR: string;
		MKLROOT: string;
		DEBUGINFOD_URLS: string;
		npm_package_json: string;
		LC_TIME: string;
		BUN_INSTALL: string;
		JOURNAL_STREAM: string;
		LLAMA_MTP_PATH: string;
		KDE_FULL_SESSION: string;
		npm_config_noproxy: string;
		PATH: string;
		npm_config_node_gyp: string;
		HSA_FORCE_FINE_GRAIN_PCIE: string;
		HSA_OVERRIDE_GFX_VERSION: string;
		DBUS_SESSION_BUS_ADDRESS: string;
		npm_config_global_prefix: string;
		KDE_APPLICATIONS_AS_SCOPE: string;
		MAIL: string;
		npm_node_execpath: string;
		npm_config_engine_strict: string;
		FLATPAK_TTY_PROGRESS: string;
		LC_NUMERIC: string;
		OLDPWD: string;
		KONSOLE_DBUS_WINDOW: string;
		HIP_FORCE_WAVEFRONT_SIZE: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * This module provides access to environment variables set _dynamically_ at runtime and that are _publicly_ accessible.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Dynamic environment variables are defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`.
 * 
 * **_Public_ access:**
 * 
 * - This module _can_ be imported into client-side code
 * - **Only** variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`) are included
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 * 
 * > [!NOTE] To get correct types, environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * >
 * > ```env
 * > MY_FEATURE_FLAG=
 * > ```
 * >
 * > You can override `.env` values from the command line like so:
 * >
 * > ```sh
 * > MY_FEATURE_FLAG="enabled" npm run dev
 * > ```
 * 
 * For example, given the following runtime environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://example.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.ENVIRONMENT); // => undefined, not public
 * console.log(env.PUBLIC_BASE_URL); // => "http://example.com"
 * ```
 * 
 * ```
 * 
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
