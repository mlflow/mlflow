var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __rest = (this && this.__rest) || function (s, e) {
    var t = {};
    for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0)
        t[p] = s[p];
    if (s != null && typeof Object.getOwnPropertySymbols === "function")
        for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
            if (e.indexOf(p[i]) < 0 && Object.prototype.propertyIsEnumerable.call(s, p[i]))
                t[p[i]] = s[p[i]];
        }
    return t;
};
import React, { useState, useEffect } from 'react';
import DropdownNavbarItem from '@theme/NavbarItem/DropdownNavbarItem';
import BrowserOnly from '@docusaurus/BrowserOnly';
function getLabel(currentVersion, versions) {
    if (currentVersion === 'latest' && versions.length > 0) {
        // version list is sorted in descending order, so the first one is the latest
        return "Version: ".concat(versions[0], " (latest)");
    }
    return "Version: ".concat(currentVersion);
}
function VersionSelectorImpl(_a) {
    var _this = this;
    var _b;
    var mobile = _a.mobile, _c = _a.label, label = _c === void 0 ? 'Version' : _c, props = __rest(_a, ["mobile", "label"]);
    var _d = useState([]), versions = _d[0], setVersions = _d[1];
    var _e = useState(true), loading = _e[0], setLoading = _e[1];
    var versionsUrl = window.location.origin + '/docs/versions.json';
    // Determine current version from URL or default to latest
    var docPath = window.location.pathname;
    var currentVersion = (_b = docPath.match(/^\/docs\/([a-zA-Z0-9.]+)/)) === null || _b === void 0 ? void 0 : _b[1];
    var versionItems = versions === null || versions === void 0 ? void 0 : versions.map(function (version) { return ({
        type: 'default',
        label: version,
        to: window.location.origin + "/docs/".concat(version, "/"),
        target: '_self',
    }); });
    useEffect(function () {
        var fetchVersions = function () { return __awaiter(_this, void 0, void 0, function () {
            var response, data, error_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 3, 4, 5]);
                        return [4 /*yield*/, fetch(versionsUrl)];
                    case 1:
                        response = _a.sent();
                        return [4 /*yield*/, response.json()];
                    case 2:
                        data = _a.sent();
                        if (data['versions'] != null) {
                            setVersions(data['versions']);
                        }
                        return [3 /*break*/, 5];
                    case 3:
                        error_1 = _a.sent();
                        return [3 /*break*/, 5];
                    case 4:
                        setLoading(false);
                        return [7 /*endfinally*/];
                    case 5: return [2 /*return*/];
                }
            });
        }); };
        fetchVersions();
    }, [versionsUrl]);
    if (loading || versions == null || versions.length === 0) {
        return null;
    }
    return (<DropdownNavbarItem {...props} mobile={mobile} label={getLabel(currentVersion, versions)} items={versionItems}/>);
}
export default function VersionSelector(props) {
    return <BrowserOnly>{function () { return <VersionSelectorImpl {...props}/>; }}</BrowserOnly>;
}
