import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// jsdom has no layout: the demo scrolls the thread into view.
Element.prototype.scrollIntoView = function scrollIntoView() {};

afterEach(() => cleanup());
